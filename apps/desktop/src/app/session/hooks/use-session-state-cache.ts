import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { preserveLocalAssistantErrors } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $busy, $messages, noteSessionActivity, setSessionAttention, setSessionWorking } from '@/store/session'

import type { ClientSessionState } from '../../types'

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
    busyRef.current = busy
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

    setMessages(preserveLocalAssistantErrors(pending.state.messages, $messages.get()))
    setBusy(pending.state.busy)
    busyRef.current = pending.state.busy
    setAwaitingResponse(pending.state.awaitingResponse)
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
