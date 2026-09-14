import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import type { ChatMessage } from '@/lib/chat-messages'
import { cn } from '@/lib/utils'
import { type ComposerAttachmentScope, createComposerAttachmentScope } from '@/store/composer'
import { requestGatewayForAgent, retainGatewayForAgent } from '@/store/gateway'
import { setSessionOwnerHint } from '@/store/session'
import {
  $sessionStates,
  $sessionTileDelegateRevision,
  retainForegroundSessionSurface,
  sessionTileDelegate
} from '@/store/session-states'
import type { SessionCreateResponse } from '@/types/hermes'

import { requestComposerFocus } from './composer/focus'
import { type SessionView } from './session-view'
import { lastVisibleMessageIsUser } from './thread-loading'

// Keep the public SDK controller/types lightweight. Importing @hermes/plugin-sdk
// must not eagerly pull the complete transcript renderer (or its app stores)
// into plugins/tests that never mount a chat panel. The real native surface is
// loaded only when a consumer renders NativeChatPanel.
const SessionChatSurface = lazy(async () => {
  const module = await import('./session-tile')

  return { default: module.SessionChatSurface }
})

export interface NativeChatBinding {
  route: NativeChatProfileRoute
  /** Live runtime minted for a fresh binding. Callers should persist the whole
   * binding, but only the stored id is durable across Desktop/backend restarts;
   * the panel automatically resumes when this runtime is stale. */
  runtimeSessionId?: string
  storedSessionId: string
}

/** Connection-qualified plugin route. Unlike core's low-level owner hint, an
 * embedded plugin chat is never allowed to rely on an ambient/partial route. */
export interface NativeChatProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  profile: string
  targetProfile: string
}

export interface CreateNativeChatSessionOptions {
  cwd?: null | string
  /** Plugin conversations are private companion surfaces by default and should
   * not create a duplicate row in the main Sessions navigation. */
  hidden?: boolean
  route: NativeChatProfileRoute
}

const MAX_PENDING_NATIVE_SESSION_LEASES = 64
const pendingNativeSessionLeases = new Map<string, () => void>()
const MAX_NATIVE_ATTACHMENT_SCOPES = 64
const nativeAttachmentScopes = new Map<string, ComposerAttachmentScope>()

function normalizeRoute(route: NativeChatProfileRoute): NativeChatProfileRoute {
  const connectionId = route.connectionId.trim()
  const profile = route.profile.trim()
  const targetProfile = route.targetProfile.trim()

  if (!connectionId || !profile || !targetProfile || (route.mode !== 'local' && route.mode !== 'remote')) {
    throw new Error('Native chat requires an exact connectionId + mode + profile + targetProfile route.')
  }

  return {
    connectionId,
    mode: route.mode,
    profile,
    targetProfile
  }
}

function nativeBindingKey(route: NativeChatProfileRoute, storedSessionId: string): string {
  return JSON.stringify([
    route.connectionId,
    route.mode,
    route.profile,
    route.targetProfile,
    storedSessionId
  ])
}

function nativeAttachmentScope(route: NativeChatProfileRoute, storedSessionId: string): ComposerAttachmentScope {
  const key = nativeBindingKey(route, storedSessionId)
  const existing = nativeAttachmentScopes.get(key)

  if (existing) {
    // Refresh bounded LRU order on reopen.
    nativeAttachmentScopes.delete(key)
    nativeAttachmentScopes.set(key, existing)

    return existing
  }

  const created = createComposerAttachmentScope()
  nativeAttachmentScopes.set(key, created)

  while (nativeAttachmentScopes.size > MAX_NATIVE_ATTACHMENT_SCOPES) {
    const oldest = nativeAttachmentScopes.keys().next().value as string | undefined

    if (!oldest) {
      break
    }

    nativeAttachmentScopes.delete(oldest)
  }

  return created
}

function rememberPendingNativeSessionLease(storedSessionId: string, release: () => void): void {
  pendingNativeSessionLeases.get(storedSessionId)?.()
  pendingNativeSessionLeases.delete(storedSessionId)
  pendingNativeSessionLeases.set(storedSessionId, release)

  while (pendingNativeSessionLeases.size > MAX_PENDING_NATIVE_SESSION_LEASES) {
    const oldest = pendingNativeSessionLeases.keys().next().value as string | undefined

    if (!oldest) {
      break
    }

    pendingNativeSessionLeases.get(oldest)?.()
    pendingNativeSessionLeases.delete(oldest)
  }
}

function releasePendingNativeSessionLease(storedSessionId: string): void {
  const release = pendingNativeSessionLeases.get(storedSessionId)

  if (!release) {
    return
  }

  pendingNativeSessionLeases.delete(storedSessionId)
  release()
}

/** @internal Tests. */
export function _resetNativeChatSessionLeasesForTests(): void {
  for (const release of pendingNativeSessionLeases.values()) {
    release()
  }

  pendingNativeSessionLeases.clear()
  nativeAttachmentScopes.clear()
}

/** @internal Tests. */
export const _nativeChatAttachmentScopeForTests = nativeAttachmentScope

/** Create a normal Hermes session on one exact profile route without selecting,
 * navigating to, or adding a layout pane for it. The route lease intentionally
 * survives until the first durable message row: an unused new backend session
 * exists only in memory, and Mail/Studio must be able to close and reopen their
 * side panel without losing an unsent native composer draft. */
export async function createNativeChatSession(
  options: CreateNativeChatSessionOptions
): Promise<NativeChatBinding> {
  const route = normalizeRoute(options.route)
  const releaseRoute = await retainGatewayForAgent(route.connectionId, route.profile)
  let keepRoute = false

  try {
    const created = await requestGatewayForAgent<SessionCreateResponse>(
      route.connectionId,
      route.profile,
      'session.create',
      {
        cols: 96,
        source: 'desktop',
        profile: route.targetProfile || route.profile,
        hidden: options.hidden ?? true,
        ...(typeof options.cwd === 'string' && options.cwd.trim() ? { cwd: options.cwd.trim() } : {})
      }
    )

    const storedSessionId = created.stored_session_id?.trim()

    if (!storedSessionId) {
      await requestGatewayForAgent(route.connectionId, route.profile, 'session.close', {
        session_id: created.session_id
      }).catch(() => undefined)
      throw new Error('Hermes created a native chat runtime without a stored session id.')
    }

    const delegate = sessionTileDelegate()

    if (!delegate?.bindCreatedSession) {
      await requestGatewayForAgent(route.connectionId, route.profile, 'session.close', {
        session_id: created.session_id
      }).catch(() => undefined)
      throw new Error('Hermes native chat session wiring is not ready.')
    }

    setSessionOwnerHint(storedSessionId, route)
    const runtimeSessionId = delegate.bindCreatedSession(created, storedSessionId)
    rememberPendingNativeSessionLease(storedSessionId, releaseRoute)
    keepRoute = true

    return { route, runtimeSessionId, storedSessionId }
  } finally {
    if (!keepRoute) {
      releaseRoute()
    }
  }
}

const NO_MESSAGES: ChatMessage[] = []

function buildNativeView(storedSessionId: string, $runtimeId: ReturnType<typeof atom<null | string>>): SessionView {
  const $state = computed([$runtimeId, $sessionStates], (runtimeId, states) =>
    runtimeId ? states[runtimeId] : undefined
  )

  const $messages = computed($state, state => state?.messages ?? NO_MESSAGES)

  return {
    kind: 'tile',
    $awaitingResponse: computed($state, state => Boolean(state?.awaitingResponse)),
    $busy: computed($state, state => Boolean(state?.busy)),
    $cwd: computed($state, state => state?.cwd ?? ''),
    $fast: computed($state, state => Boolean(state?.fast)),
    $lastVisibleIsUser: computed($messages, lastVisibleMessageIsUser),
    $messages,
    $messagesEmpty: computed($messages, messages => messages.length === 0),
    $model: computed($state, state => state?.model ?? ''),
    $provider: computed($state, state => state?.provider ?? ''),
    $reasoningEffort: computed($state, state => state?.reasoningEffort ?? ''),
    $runtimeId,
    $storedId: atom(storedSessionId),
    $turnStartedAt: computed($state, state => state?.turnStartedAt ?? null)
  }
}

export interface NativeChatPanelProps {
  binding: NativeChatBinding
  className?: string
  /** Increment to focus this panel's native composer without changing Hub
   * navigation or the globally focused layout session. */
  focusRequest?: number
}

/** Native Hermes transcript + composer embedded inside a plugin-owned surface.
 * It is a session surface, not a second chat client: resume/actions/events all
 * use the existing shared session state, exact-route request router and native
 * ChatView/ChatBar implementation. */
export function NativeChatPanel({ binding, className, focusRequest = 0 }: NativeChatPanelProps) {
  const {
    connectionId: routeConnectionId,
    mode: routeMode,
    profile: routeProfile,
    targetProfile: routeTargetProfile
  } = binding.route

  // Keyed by the route's PRIMITIVE fields, not the route object's identity: a
  // plugin's inline render produces an equivalent binding on every render, and
  // an identity-keyed route would re-register the foreground lease below on
  // each one.
  const route = useMemo(
    () =>
      normalizeRoute({
        connectionId: routeConnectionId,
        mode: routeMode,
        profile: routeProfile,
        targetProfile: routeTargetProfile
      }),
    [routeConnectionId, routeMode, routeProfile, routeTargetProfile]
  )

  const storedSessionId = binding.storedSessionId.trim()
  const target = `native:${storedSessionId}`

  const attachments = useMemo(
    () => nativeAttachmentScope(route, storedSessionId),
    [route, storedSessionId]
  )

  const $runtimeId = useMemo(
    () => {
      const candidate = binding.runtimeSessionId?.trim() || ''
      const canReuseFreshRuntime = Boolean(candidate && pendingNativeSessionLeases.has(storedSessionId))
      const hasSharedState = Boolean(candidate && $sessionStates.get()[candidate])

      return atom<null | string>(canReuseFreshRuntime || hasSharedState ? candidate : null)
    },
    [binding.runtimeSessionId, storedSessionId]
  )

  const view = useMemo(() => buildNativeView(storedSessionId, $runtimeId), [$runtimeId, storedSessionId])
  const runtimeId = useStore($runtimeId)
  const delegateRevision = useStore($sessionTileDelegateRevision)
  const [error, setError] = useState<string | null>(null)
  const [retryRevision, setRetryRevision] = useState(0)

  useEffect(() => {
    if (!runtimeId || $sessionStates.get()[runtimeId]) {
      return
    }

    // A plugin can close/reopen before its first turn. The binding still owns
    // the live in-memory runtime even if its idle presentation slice was
    // evicted while the panel was unmounted; re-seed that shared slice instead
    // of issuing session.resume for a conversation that is not durable yet.
    sessionTileDelegate()?.updateSession(runtimeId, state => ({ ...state }), storedSessionId)
  }, [delegateRevision, runtimeId, storedSessionId])

  useEffect(() => {
    if (!storedSessionId) {
      setError('Native chat binding has no stored session id.')

      return
    }

    setSessionOwnerHint(storedSessionId, route)

    return retainForegroundSessionSurface(route, runtimeId)
  }, [route, runtimeId, storedSessionId])

  // The resume is keyed to the exact binding it belongs to: a panel that is
  // re-bound while an earlier resume is still in flight must start the NEW
  // binding's resume instead of leaving itself on "Connecting" forever. A stale
  // resume that settles later is dropped by its own cleanup.
  const resumeKey = nativeBindingKey(route, storedSessionId)
  const resumingKeyRef = useRef<null | string>(null)
  // eslint-disable-next-line no-restricted-syntax -- in-flight resume latch keyed to the binding, not an atom mirror
  useEffect(() => {
    // The shared delegate is installed by Desktop's normal session wiring. It
    // performs exact-owner resume, transcript hydration, approval restoration
    // and shared state publication without layout/navigation side effects.
    if (!storedSessionId || $runtimeId.get() || resumingKeyRef.current === resumeKey || !sessionTileDelegate()) {
      return
    }

    let cancelled = false

    resumingKeyRef.current = resumeKey
    setError(null)

    void sessionTileDelegate()!
      .resumeTile(storedSessionId, { refreshTranscript: true })
      .then(resumedRuntimeId => {
        if (!cancelled) {
          $runtimeId.set(resumedRuntimeId)
        }
      })
      .catch(reason => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason))
        }
      })
      .finally(() => {
        if (resumingKeyRef.current === resumeKey) {
          resumingKeyRef.current = null
        }
      })

    return () => {
      cancelled = true
    }
  }, [$runtimeId, delegateRevision, resumeKey, retryRevision, storedSessionId])

  // A durable row means the backend session now survives socket pruning. Until
  // then the creation lease stays alive even if the plugin temporarily unmounts
  // its side panel, preserving unsent close/reopen semantics.
  useEffect(
    () =>
      view.$messages.subscribe(messages => {
        if (messages.some(message => typeof message.rowId === 'number')) {
          releasePendingNativeSessionLease(storedSessionId)
        }
      }),
    [storedSessionId, view.$messages]
  )

  useEffect(() => {
    if (runtimeId && focusRequest > 0) {
      requestComposerFocus(target)
    }
  }, [focusRequest, runtimeId, target])

  const onRuntimeRecovered = useCallback((nextRuntimeId: string) => $runtimeId.set(nextRuntimeId), [$runtimeId])

  const retry = useCallback(() => {
    $runtimeId.set(null)
    setError(null)
    setRetryRevision(value => value + 1)
  }, [$runtimeId])

  if (error) {
    return (
      <div className={cn('grid h-full min-h-0 place-items-center p-4', className)}>
        <ErrorState description={error} title="Hermes conversation unavailable">
          <Button onClick={retry} size="sm" variant="outline">
            Retry
          </Button>
        </ErrorState>
      </div>
    )
  }

  if (!runtimeId) {
    return (
      <div aria-busy="true" className={cn('grid h-full min-h-0 place-items-center text-sm text-(--ui-text-tertiary)', className)}>
        Connecting to Hermes…
      </div>
    )
  }

  return (
    <Suspense
      fallback={
        <div aria-busy="true" className={cn('grid h-full min-h-0 place-items-center text-sm text-(--ui-text-tertiary)', className)}>
          Loading Hermes…
        </div>
      }
    >
      <SessionChatSurface
        attachmentScope={attachments}
        className={className}
        forceFocused
        listSessionOnFirstSend={false}
        onRetryResume={retry}
        onRuntimeRecovered={onRuntimeRecovered}
        ownerRoute={route}
        retainAttachmentsAcrossUnmount
        runtimeId={runtimeId}
        scopeTarget={target}
        sessionAnchorOverride={null}
        storedSessionId={storedSessionId}
        view={view}
      />
    </Suspense>
  )
}
