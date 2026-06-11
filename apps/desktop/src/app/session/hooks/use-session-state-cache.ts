import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { preserveLocalAssistantErrors } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  $busy,
  $messages,
  noteSessionActivity,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setSessionAttention,
  setSessionWorking,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'

import type { ClientSessionState } from '../../types'

// Shallow per-message identity check. When a flush carries no transcript
// changes, `preserveLocalAssistantErrors` returns the same message objects in
// the same order, so reference equality per slot is enough to detect "nothing
// to publish" and avoid a needless `$messages` churn.
function sameMessageList(a: ChatMessage[], b: ChatMessage[]): boolean {
  if (a === b) {
    return true
  }

  if (a.length !== b.length) {
    return false
  }

  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) {
      return false
    }
  }

  return true
}

interface SessionStateCacheOptions {
  activeSessionId: string | null
  busyRef: MutableRefObject<boolean>
  selectedStoredSessionId: string | null
  setAwaitingResponse: (awaiting: boolean) => void
  setBusy: (busy: boolean) => void
  setMessages: (messages: ChatMessage[]) => void
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
      if (storedSessionId !== undefined) {
        const previousStoredSessionId = existing.storedSessionId
        existing.storedSessionId = storedSessionId

        if (storedSessionId) {
          runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)

          if (existing.busy) {
            setSessionWorking(storedSessionId, true)
          }
        }

        if (previousStoredSessionId && previousStoredSessionId !== storedSessionId) {
          setSessionWorking(previousStoredSessionId, false)
        }
      }

      return existing
    }

    const created = createClientSessionState(storedSessionId ?? null)
    sessionStateByRuntimeIdRef.current.set(sessionId, created)

    if (storedSessionId) {
      runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)
    }

    return created
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
    // the merged result is content-identical to what's already on screen.
    const currentMessages = $messages.get()
    const nextMessages = preserveLocalAssistantErrors(pending.state.messages, currentMessages)

    if (!sameMessageList(nextMessages, currentMessages)) {
      setMessages(nextMessages)
    }

    setCurrentModel(pending.state.model)
    setCurrentProvider(pending.state.provider)
    setCurrentReasoningEffort(pending.state.reasoningEffort)
    setCurrentServiceTier(pending.state.serviceTier)
    setCurrentFastMode(pending.state.fast)
    setYoloActive(pending.state.yolo)
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

      if (previous.storedSessionId !== next.storedSessionId || !next.busy) {
        setSessionWorking(previous.storedSessionId, false)
      }

      if (previous.storedSessionId !== next.storedSessionId || !next.needsInput) {
        setSessionAttention(previous.storedSessionId, false)
      }

      setSessionWorking(next.storedSessionId, next.busy)
      setSessionAttention(next.storedSessionId, next.needsInput)

      // Every state update is effectively a "still alive" heartbeat for
      // streaming events. The session-store watchdog uses this to keep the
      // working flag alive during long-running turns and to clear it once
      // the stream goes silent.
      if (next.busy) {
        noteSessionActivity(next.storedSessionId)
      }

      syncSessionStateToView(sessionId, next)

      return next
    },
    [ensureSessionState, syncSessionStateToView]
  )

  return {
    activeSessionIdRef,
    ensureSessionState,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  }
}
