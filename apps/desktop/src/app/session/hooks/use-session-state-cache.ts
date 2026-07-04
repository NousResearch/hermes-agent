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
  onSessionWatchdogClear,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setSessionAttention,
  setSessionWorking,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import type { PaneId } from '@/store/split'

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

/**
 * The projection surface one registered pane exposes to the cache: its view
 * atoms' setters, the messages atom it publishes into, and the refs its
 * staging gate reads. `PaneSessionView` (app/chat/pane-view.ts) satisfies this
 * structurally; the MAIN pane's entry is assembled from the hook options so
 * the constructor contract (setMessages/setBusy/setAwaitingResponse/busyRef)
 * is unchanged.
 */
export interface RegisteredSessionView {
  paneId: PaneId
  $messages: { get: () => ChatMessage[] }
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  /** Runtime id whose transcript currently occupies the view's `$messages` —
   *  lets the flush tell a same-session refresh from a thread switch. */
  viewSessionIdRef: MutableRefObject<string | null>
  setMessages: (messages: ChatMessage[]) => void
  setBusy: (busy: boolean) => void
  setAwaitingResponse: (awaiting: boolean) => void
  setTurnStartedAt: (turnStartedAt: number | null) => void
  setCurrentModel: (model: string) => void
  setCurrentProvider: (provider: string) => void
  setCurrentReasoningEffort: (effort: string) => void
  setCurrentServiceTier: (tier: string) => void
  setCurrentFastMode: (fast: boolean) => void
  setYoloActive: (yolo: boolean) => void
  setCurrentPersonality: (personality: string) => void
}

// One registered pane = one staging slot + one RAF handle. Per-pane batching
// keeps a busy background pane's flush cadence independent of the active one.
interface RegisteredPaneView {
  view: RegisteredSessionView
  pending: { sessionId: string; state: ClientSessionState } | null
  raf: number | null
}

function syncRuntimeMetadataToView(view: RegisteredSessionView, state: ClientSessionState) {
  view.setCurrentModel(state.model ?? '')
  view.setCurrentProvider(state.provider ?? '')
  view.setCurrentReasoningEffort(state.reasoningEffort ?? '')
  view.setCurrentServiceTier(state.serviceTier ?? '')
  view.setCurrentFastMode(state.fast ?? false)
  view.setYoloActive(state.yolo ?? false)
  view.setCurrentPersonality(state.personality ?? '')
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
  // Runtime id whose transcript currently occupies the MAIN `$messages` view.
  const viewSessionIdRef = useRef<string | null>(null)

  // Latest options snapshot, refreshed every render, so the (stable) main
  // pane-view entry below always flushes through the caller's current setters —
  // exactly like the old useCallback deps did.
  const mainOptionsRef = useRef({ busyRef, setAwaitingResponse, setBusy, setMessages })
  mainOptionsRef.current = { busyRef, setAwaitingResponse, setBusy, setMessages }

  // The MAIN pane registered from the hook options: reads the global $messages,
  // publishes through the option setters, and writes the global metadata
  // singletons — line-for-line the old single-view projection.
  const mainPaneViewRef = useRef<RegisteredSessionView | null>(null)

  if (mainPaneViewRef.current === null) {
    mainPaneViewRef.current = {
      paneId: 'main',
      $messages,
      activeSessionIdRef,
      get busyRef() {
        return mainOptionsRef.current.busyRef
      },
      viewSessionIdRef,
      setMessages: messages => mainOptionsRef.current.setMessages(messages),
      setBusy: value => mainOptionsRef.current.setBusy(value),
      setAwaitingResponse: value => mainOptionsRef.current.setAwaitingResponse(value),
      setTurnStartedAt,
      setCurrentModel,
      setCurrentProvider,
      setCurrentReasoningEffort,
      setCurrentServiceTier,
      setCurrentFastMode,
      setYoloActive,
      setCurrentPersonality
    }
  }

  const paneViewsRef = useRef<Map<PaneId, RegisteredPaneView> | null>(null)

  if (paneViewsRef.current === null) {
    paneViewsRef.current = new Map([['main', { pending: null, raf: null, view: mainPaneViewRef.current }]])
  }

  const paneViews = paneViewsRef as MutableRefObject<Map<PaneId, RegisteredPaneView>>

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

  const flushPendingPaneState = useCallback((entry: RegisteredPaneView) => {
    const pending = entry.pending
    entry.pending = null

    if (!pending || pending.sessionId !== entry.view.activeSessionIdRef.current) {
      return
    }

    // `preserveLocalAssistantErrors` always returns a fresh array, so publishing
    // it unconditionally puts a new `$messages` reference on the store every
    // flush — including the periodic `session.info` heartbeats that don't touch
    // the transcript. That churns ChatView → runtimeMessageRepository → the
    // assistant-ui runtime → the virtualizer, which re-measures and visibly
    // jerks the scroll position while the user is reading. Skip the publish when
    // the merged result is content-identical to what's already on screen.
    const currentMessages = entry.view.$messages.get()

    // On a thread switch the view still holds the *previous* thread, so
    // preserving its local errors would graft that thread's failed turn (e.g.
    // an out-of-funds error) onto this one — then cascade it everywhere as the
    // polluted view becomes the next switch's baseline. Only carry errors
    // across a same-session refresh; our cached state already keeps its own.
    const nextMessages =
      entry.view.viewSessionIdRef.current === pending.sessionId
        ? preserveLocalAssistantErrors(pending.state.messages, currentMessages)
        : pending.state.messages

    if (!sameMessageList(nextMessages, currentMessages)) {
      entry.view.setMessages(nextMessages)
    }

    entry.view.viewSessionIdRef.current = pending.sessionId

    syncRuntimeMetadataToView(entry.view, pending.state)
    entry.view.setBusy(pending.state.busy)
    setMutableRef(entry.view.busyRef, pending.state.busy)
    entry.view.setAwaitingResponse(pending.state.awaitingResponse)
    // Mirror the focused session's per-session turn clock into the atom the
    // statusbar timer reads. Keeps a backgrounded turn's elapsed time intact
    // on focus instead of zeroing it (the "timer restarts" bug).
    entry.view.setTurnStartedAt(pending.state.turnStartedAt)
  }, [])

  const syncPaneStateToView = useCallback(
    (entry: RegisteredPaneView, sessionId: string, state: ClientSessionState) => {
      // Only the pane's currently-viewed session may stage into its view. A
      // background session (e.g. one still busy and emitting stream / error
      // updates after the user toggled away) must update its own cache entry
      // but never a view — otherwise its messages clobber the foreground
      // transcript and appear to "bleed" into every other session. The flush
      // above also re-checks the pane's active id, but staging here is what
      // prevents a background write from overwriting an already-pending
      // foreground write within the same animation frame (only one RAF is
      // scheduled per pane, so the last `pending` writer would otherwise win).
      if (sessionId !== entry.view.activeSessionIdRef.current) {
        return
      }

      syncRuntimeMetadataToView(entry.view, state)
      entry.pending = { sessionId, state }

      // Terminal / attention transitions (turn finished, error, or the agent is
      // now waiting on the user) MUST reach the view immediately. Electron
      // throttles `requestAnimationFrame` to ~0 while the window is
      // backgrounded, occluded, or unfocused, so an RAF-deferred flush can be
      // stranded in `pending` indefinitely — that's the "new chat stuck on
      // Thinking until I refocus / F5" bug. Flush these synchronously
      // (cancelling any in-flight RAF, since we're about to publish the latest
      // state anyway). The plain busy heartbeat stays RAF-batched: that
      // coalescing exists only to keep periodic `session.info` updates from
      // churning `$messages` and jerking the scroll position while reading.
      const isCriticalTransition = !state.busy || state.needsInput

      if (isCriticalTransition) {
        if (entry.raf !== null && typeof window !== 'undefined') {
          window.cancelAnimationFrame(entry.raf)
          entry.raf = null
        }

        flushPendingPaneState(entry)

        return
      }

      if (entry.raf !== null) {
        return
      }

      if (typeof window === 'undefined') {
        flushPendingPaneState(entry)

        return
      }

      entry.raf = window.requestAnimationFrame(() => {
        entry.raf = null
        flushPendingPaneState(entry)
      })
    },
    [flushPendingPaneState]
  )

  // Fan a session's fresh truth out to every registered pane view. With only
  // the main pane registered (single-pane mode) this is exactly the old
  // single-view staging + flush.
  const syncSessionStateToView = useCallback(
    (sessionId: string, state: ClientSessionState) => {
      for (const entry of paneViews.current.values()) {
        syncPaneStateToView(entry, sessionId, state)
      }
    },
    [paneViews, syncPaneStateToView]
  )

  /** Mount a pane's view for fan-out. Returns the unregister disposer —
   *  SplitChatPane registers on mount and unregisters on close, after which
   *  its session keeps streaming in the truth layer like any background one. */
  const registerPaneView = useCallback(
    (view: RegisteredSessionView) => {
      const entry: RegisteredPaneView = { pending: null, raf: null, view }
      paneViews.current.set(view.paneId, entry)

      return () => {
        if (entry.raf !== null && typeof window !== 'undefined') {
          window.cancelAnimationFrame(entry.raf)
          entry.raf = null
        }

        if (paneViews.current.get(view.paneId) === entry) {
          paneViews.current.delete(view.paneId)
        }
      }
    },
    [paneViews]
  )

  /** Re-project a pane's cached session truth into its view synchronously —
   *  the warm repaint used on pane activation and split-open (mirrors the
   *  warm-cache repaint in resumeSession). No-op when the pane isn't
   *  registered or holds no cached session. */
  const publishPaneState = useCallback(
    (paneId: PaneId) => {
      const entry = paneViews.current.get(paneId)
      const runtimeId = entry?.view.activeSessionIdRef.current
      const state = runtimeId ? sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined

      if (!entry || !runtimeId || !state) {
        return
      }

      if (entry.raf !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(entry.raf)
        entry.raf = null
      }

      entry.pending = { sessionId: runtimeId, state }
      flushPendingPaneState(entry)
    },
    [flushPendingPaneState, paneViews]
  )

  useEffect(
    () => () => {
      for (const entry of paneViews.current.values()) {
        if (entry.raf !== null && typeof window !== 'undefined') {
          window.cancelAnimationFrame(entry.raf)
          entry.raf = null
        }
      }
    },
    [paneViews]
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

  // When the store watchdog force-clears a stuck session (8 min of stream
  // silence — a hung or looping turn that never delivered its terminal event),
  // also drop that session's busy/awaiting flags here. Clearing the sidebar dot
  // alone leaves the composer wedged on "Thinking"/Stop; updateSessionState
  // re-syncs `$busy` when the healed session is the one on screen.
  useEffect(
    () =>
      onSessionWatchdogClear(storedSessionId => {
        const runtimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        const state = runtimeId ? sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined

        if (!runtimeId || !state?.busy) {
          return
        }

        updateSessionState(runtimeId, current => ({
          ...current,
          awaitingResponse: false,
          busy: false,
          needsInput: false
        }))
      }),
    [updateSessionState]
  )

  return {
    activeSessionIdRef,
    ensureSessionState,
    publishPaneState,
    registerPaneView,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  }
}
