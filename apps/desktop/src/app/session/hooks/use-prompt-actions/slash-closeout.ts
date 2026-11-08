import type { MutableRefObject } from 'react'

import { prepareDefaultNewSession } from '@/app/session/new-session-route'
import { createClientSessionState } from '@/lib/chat-runtime'
import { closeoutCommand } from '@/lib/session-closeout-command'
import { notifyError, RECOVERY_ACTIONS } from '@/store/notifications'
import { $sessions, resolveComposerSessionKey, setSessionOwnerHint } from '@/store/session'
import { captureCloseoutTarget, type SessionCloseoutReceipt } from '@/store/session-closeout'
import { $focusedStoredSessionId } from '@/store/session-focus'
import {
  $focusedRuntimeId,
  $sessionTiles,
  focusOpenSession,
  openSessionTile,
  patchSessionTile,
  publishSessionState,
  sessionTileDelegate,
  storedSessionIdForRuntimeId
} from '@/store/session-states'
import { captureSessionViewer, releaseSessionViewer } from '@/store/session-viewer-lifecycle'
import type { SessionCreateResponse } from '@/types/hermes'

import { captureSubmissionDestination } from './submission-destination'
import type { GatewayRequest, SubmitTextOptions } from './utils'

function assertCreationCurrent(created: SessionCreateResponse, target: ReturnType<typeof captureCloseoutTarget>) {
  if (!target.isCurrent()) {
    captureSessionViewer(created.session_id, target.destination.owner)
    releaseSessionViewer(created.session_id, createClientSessionState(created.stored_session_id))
    target.assertCurrent()
  }
}

async function createIndependentSession(
  receipt: SessionCloseoutReceipt,
  target: ReturnType<typeof captureCloseoutTarget>,
  title?: string,
  workspaceMode?: string
) {
  const { destination } = target
  target.assertCurrent()

  const created = await target.request<SessionCreateResponse>('session.create', {
    source: 'desktop',
    cols: 96,
    profile: receipt.old.profile,
    request_id: receipt.next_create_request_id
  })

  const fresh = created.stored_session_id

  assertCreationCurrent(created, target)

  if (
    !fresh ||
    fresh === receipt.old.logical_session_id ||
    fresh === receipt.old.stored_session_id ||
    created.session_id === receipt.old.runtime_id ||
    created.session_id === receipt.old.logical_session_id
  ) {
    throw new Error('The backend did not create an independent conversation; the preserved transcript is retained')
  }

  if (destination.owner && typeof destination.owner === 'object') {
    setSessionOwnerHint(fresh, destination.owner)
  }

  if (title && workspaceMode !== 'bots') {
    await target.request('session.title', {
      session_id: created.session_id,
      profile: receipt.old.profile,
      title
    }).finally(() => assertCreationCurrent(created, target))
  }

  return { created, fresh }
}

interface CloseoutCommandContext {
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  getRoutedStoredSessionId: () => string | null
  requestGateway: GatewayRequest
  startFreshSessionDraft: () => void
}

export async function runCloseoutCommand(
  text: string,
  options: SubmitTextOptions | undefined,
  context: CloseoutCommandContext
): Promise<boolean | undefined> {
  const command = closeoutCommand(text)

  if (!command) {
    return undefined
  }

  if (options?.attachments?.length) {
    return false
  }

  const stored =
    options?.storedSessionId ??
    (options?.sessionId ? storedSessionIdForRuntimeId(options.sessionId) : null) ??
    context.getRoutedStoredSessionId() ??
    context.selectedStoredSessionIdRef.current

  if (options?.composerScope !== undefined && options.composerScope !== resolveComposerSessionKey(stored, $sessions.get())) {
    return false
  }

  if (!stored) {
    return command.intent === 'archive' ? true : undefined
  }

  const destination = options?.destination ?? captureSubmissionDestination(stored, context.requestGateway)
  const tile = $sessionTiles.get().find(candidate => candidate.storedSessionId === stored)
  const focused = $focusedStoredSessionId.get() ?? context.selectedStoredSessionIdRef.current
  const focusedRuntime = $focusedRuntimeId.get()
  const hash = window.location.hash

  const isCurrent = () =>
    window.location.hash === hash &&
    (focusedRuntime
      ? $focusedRuntimeId.get() === focusedRuntime
      : ($focusedStoredSessionId.get() ?? context.selectedStoredSessionIdRef.current) === focused) &&
    focused === stored

  try {
    const delegate = sessionTileDelegate()

    if (!delegate) {
      throw new Error('Conversation controls are not ready; retry from the original chat')
    }

    const target = captureCloseoutTarget(stored, context.requestGateway, { destination })

    const receipt = await delegate.archiveSession(stored, {
      intent: command.intent,
      destination,
      requestId: options?.submission_id,
      ...(command.intent === 'new' ? { deferUi: true } : {})
    })

    if (!receipt) {
      return false
    }

    if (command.intent === 'archive') {
      return true
    }

    const { created, fresh } = await createIndependentSession(receipt, target, command.title, tile?.workspaceMode)
    assertCreationCurrent(created, target)
    const show = isCurrent()

    const applied = await delegate.archiveSession(stored, {
      intent: 'new',
      destination,
      receipt,
      preserveView: !show,
      onApplied: () => {
        if (!show) {
          captureSessionViewer(created.session_id, destination.owner)
          releaseSessionViewer(created.session_id, createClientSessionState(fresh))

          return
        }

        openSessionTile(fresh, 'center', undefined, undefined, {
          workspaceMode: tile?.workspaceMode ?? 'sessions',
          workspaceOwnerKey: tile?.workspaceOwnerKey,
          ...(destination.owner && typeof destination.owner === 'object' ? { ownerRoute: destination.owner } : {})
        })
        patchSessionTile(fresh, {
          runtimeId: created.session_id,
          ...(typeof destination.owner === 'string' ? { ownerProfile: destination.owner } : {})
        })
        publishSessionState(created.session_id, createClientSessionState(fresh))
        focusOpenSession(fresh, {
          workspaceMode: tile?.workspaceMode ?? 'sessions',
          workspaceOwnerKey: tile?.workspaceOwnerKey
        })
      }
    })

    return Boolean(applied)
  } catch (error) {
    notifyError(error, 'Conversation closeout failed', { action: RECOVERY_ACTIONS.openGateways() })

    return false
  }
}

export function closeoutCommandHandlers(
  context: CloseoutCommandContext,
  options: SubmitTextOptions | undefined,
  rejected: () => void
) {
  return {
    new: async (ctx: { name: string; command: string }) => {
      const handled = ctx.name === 'new' ? await runCloseoutCommand(ctx.command, options, context) : undefined

      if (handled !== undefined) {
        if (!handled) {
          rejected()
        }

        return
      }

      prepareDefaultNewSession()
      context.startFreshSessionDraft()
    },
    archive: async (ctx: { command: string }) => {
      if ((await runCloseoutCommand(ctx.command, options, context)) === false) {
        rejected()
      }
    }
  }
}
