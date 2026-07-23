import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { preserveLocalAssistantErrors } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  $busy,
  $messages,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { publishSessionState } from '@/store/session-states'

import type { ClientSessionState } from '../../types'

import { chatMessageArraysEquivalent } from './use-session-actions/utils'

interface SessionStateCacheOptions {
  activeSessionId: string | null
  busyRef: MutableRefObject<boolean>
  selectedStoredSessionId: string | null
  setAwaitingResponse: (awaiting: boolean) => void
  setBusy: (busy: boolean) => void
  setMessages: (messages: ChatMessage[]) => void
}

function syncRuntimeMetadataToView(state: ClientSessionState) {
  setCurrentModel(state.model ?? '')
  setCurrentProvider(state.provider ?? '')
  setCurrentReasoningEffort(state.reasoningEffort ?? '')
  setCurrentServiceTier(state.serviceTier ?? '')
  setCurrentFastMode(state.fast ?? false)
  setYoloActive(state.yolo ?? false)
  setCurrentPersonality(state.personality ?? '')
}

export function useSessionStateCache({
  activeSessionId,
  busyRef,
  selectedStoredSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages
}: SessionStateCacheOptions) {
  const busy = useStore($busy)
  const activeSessionIdRef = useRef<string | null>(null)
  const selectedStoredSessionIdRef = useRef<string | null>(null)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const pendingViewStateRef = useRef<{ sessionId: string; state: ClientSessionState } | null>(null)
  const viewSyncRafRef = useRef<number | null>(null)
  // Runtime id whose transcript currently occupies `$messages` — lets the
  // flush below tell a same-session refresh from a thread switch.
  const viewSessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId
  }, [activeSessionId])

  useEffect(() => {
    setMutableRef(busyRef, busy)
  }, [busy, busyRef])

  useEffect(() => {
    selectedStoredSessionIdRef.current = selectedStoredSessionId
  }, [selectedStoredSessionId])

  const ensureSessionState = useCallback((sessionId: string, storedSessionId?: string | null) => {
    const existing = sessionStateByRuntimeIdRef.current.get(sessionId)

    if (existing) {
      if (storedSessionId !== undefined && storedSessionId !== existing.storedSessionId) {
        // Stored id changed (e.g. auto-compression rotated it). Create a NEW
        // state object rather than mutating in place — updateSessionState needs
        // the PREVIOUS state to detect transitions (busy→idle, id rotation).
        const updated = { ...existing, storedSessionId }

        sessionStateByRuntimeIdRef.current.set(sessionId, updated)

        // Drop the obsolete stored→runtime reverse mapping as soon as the id
        // rotates (e.g. auto-compression forks a continuation). Leaving the
        // stale key lets getRuntimeIdForStoredSession resolve the old stored id
        // to this runtime, which the compression route-follow logic relies on
        // being absent. The rotation signal itself is emitted centrally from
        // handleTransition (session-states.ts) off the published diff.
        if (existing.storedSessionId && existing.storedSessionId !== storedSessionId) {
          runtimeIdByStoredSessionIdRef.current.delete(existing.storedSessionId)
        }

        if (storedSessionId) {
          runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)
        }
      }

      return sessionStateByRuntimeIdRef.current.get(sessionId)!
    }

    const created = createClientSessionState(storedSessionId ?? null)
    sessionStateByRuntimeIdRef.current.set(sessionId, created)

    if (storedSessionId) {
      runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)
    }

    return created
  }, [])

  const resetViewSync = useCallback(() => {
    // Drop any RAF-pending transcript stage so a backgrounded turn cannot
    // repaint over the chat the user just switched to (#47709 / #47743).
    pendingViewStateRef.current = null
    viewSessionIdRef.current = null

    if (viewSyncRafRef.current !== null && typeof window !== 'undefined') {
      window.cancelAnimationFrame(viewSyncRafRef.current)
      viewSyncRafRef.current = null
    }
  }, [])

  const flushPendingViewState = useCallback(() => {
    const pending = pendingViewStateRef.current
    pendingViewStateRef.current = null

    if (!pending || pending.sessionId !== activeSessionIdRef.current) {
      return
    }

    // `preserveLocalAssistantErrors` always returns a fresh array, so publishing
    // it unconditionally puts a new `$messages` reference on the store every
    // flush — including the periodic `session.info` heartbeats that don't touch
    // the transcript. That churns ChatView → runtimeMessageRepository → the
    // assistant-ui runtime → the virtualizer, which re-measures and visibly
    // jerks the scroll position while the user is reading. Skip the publish when
    // the merged result is content-equivalent to what's already on screen.
    // Deep comparison (not just reference equality) is needed because the warm
    // resume path's `reconcileAuthoritativeMessages` creates new message objects
    // via `toChatMessages` even when the content hasn't changed — reference
    // equality would fail and cause a redundant second paint (the "warm resume
    // jitter" bug).
    const currentMessages = $messages.get()

    // On a thread switch `$messages` still holds the *previous* thread, so
    // preserving its local errors would graft that thread's failed turn (e.g.
    // an out-of-funds error) onto this one — then cascade it everywhere as the
    // polluted view becomes the next switch's baseline. Only carry errors
    // across a same-session refresh; our cached state already keeps its own.
    const nextMessages =
      viewSessionIdRef.current === pending.sessionId
        ? preserveLocalAssistantErrors(pending.state.messages, currentMessages)
        : pending.state.messages

    if (!chatMessageArraysEquivalent(nextMessages, currentMessages)) {
      setMessages(nextMessages)
    }

    viewSessionIdRef.current = pending.sessionId

    syncRuntimeMetadataToView(pending.state)
    setBusy(pending.state.busy)
    setMutableRef(busyRef, pending.state.busy)
    setAwaitingResponse(pending.state.awaitingResponse)
    // Mirror the focused session's per-session turn clock into the global
    // atom the statusbar timer reads. Keeps a backgrounded turn's elapsed
    // time intact on focus instead of zeroing it (the "timer restarts" bug).
    setTurnStartedAt(pending.state.turnStartedAt)
  }, [busyRef, setAwaitingResponse, setBusy, setMessages])

  const syncSessionStateToView = useCallback(
    (sessionId: string, state: ClientSessionState) => {
      // Only the currently-viewed session may stage into the shared `$messages`
      // view. A background session (e.g. one still busy and emitting stream /
      // error updates after the user toggled away) must update its own cache
      // entry but never the view — otherwise its messages clobber the
      // foreground transcript and appear to "bleed" into every other session.
      // The flush below also re-checks the active id, but staging here is what
      // prevents a background write from overwriting an already-pending
      // foreground write within the same animation frame (only one RAF is
      // scheduled, so the last `pendingViewStateRef` writer would otherwise win).
      if (sessionId !== activeSessionIdRef.current) {
        return
      }

      syncRuntimeMetadataToView(state)
      pendingViewStateRef.current = { sessionId, state }

      // Terminal / attention transitions (turn finished, error, or the agent is
      // now waiting on the user) MUST reach the view immediately. Electron
      // throttles `requestAnimationFrame` to ~0 while the window is
      // backgrounded, occluded, or unfocused, so an RAF-deferred flush can be
      // stranded in `pendingViewStateRef` indefinitely — that's the "new chat
      // stuck on Thinking until I refocus / F5" bug. Flush these synchronously
      // (cancelling any in-flight RAF, since we're about to publish the latest
      // state anyway). The plain busy heartbeat stays RAF-batched: that
      // coalescing exists only to keep periodic `session.info` updates from
      // churning `$messages` and jerking the scroll position while reading.
      const isCriticalTransition = !state.busy || state.needsInput

      if (isCriticalTransition) {
        if (viewSyncRafRef.current !== null && typeof window !== 'undefined') {
          window.cancelAnimationFrame(viewSyncRafRef.current)
          viewSyncRafRef.current = null
        }

        flushPendingViewState()

        return
      }

      if (viewSyncRafRef.current !== null) {
        return
      }

      if (typeof window === 'undefined') {
        flushPendingViewState()

        return
      }

      viewSyncRafRef.current = window.requestAnimationFrame(() => {
        viewSyncRafRef.current = null
        flushPendingViewState()
      })
    },
    [flushPendingViewState]
  )

  useEffect(
    () => () => {
      if (viewSyncRafRef.current !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(viewSyncRafRef.current)
        viewSyncRafRef.current = null
      }
    },
    []
  )

  const updateSessionState = useCallback(
    (
      sessionId: string,
      updater: (state: ClientSessionState) => ClientSessionState,
      storedSessionId?: string | null
    ) => {
      const previous = ensureSessionState(sessionId, storedSessionId)
      const next = updater({ ...previous, messages: previous.messages })
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      // Publishing to $sessionStates automatically fires transition side-effects
      // (watchdog, settle grace, unread marker, compression id rotation) inside
      // publishSessionState — no manual transition call needed.
      publishSessionState(sessionId, next)
      syncSessionStateToView(sessionId, next)

      return next
    },
    [ensureSessionState, syncSessionStateToView]
  )

  const getRuntimeIdForStoredSession = useCallback((storedSessionId: string): string | null => {
    const runtimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)

    if (!runtimeId) {
      return null
    }

    const runtimeState = sessionStateByRuntimeIdRef.current.get(runtimeId)

    return runtimeState?.storedSessionId === storedSessionId ? runtimeId : null
  }, [])

  return {
    activeSessionIdRef,
    ensureSessionState,
    getRuntimeIdForStoredSession,
    resetViewSync,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  }
}
