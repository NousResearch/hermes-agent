import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import type { ChatMessage } from '@/lib/chat-messages'
import { preserveLocalAssistantErrors } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { persistInFlightTurnState } from '@/lib/inflight-turn-journal'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  $activeSessionId,
  $connection,
  $messages,
  setActiveSessionStoredIdRotation,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { $sessionStates, $sessionTiles, publishSessionState, releaseSessionTranscript } from '@/store/session-states'

import type { ClientSessionState } from '../../types'
import { SessionStateCache } from '../session-state-cache'

import {
  invalidatePersistedDisplayTranscriptAuthority,
  suppressTranscriptForView
} from './use-session-actions/transcript-provenance'
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
  const busy = useStore(PRIMARY_SESSION_VIEW.$busy)
  const sessionTiles = useStore($sessionTiles)
  const activeSessionIdRef = useRef<string | null>(activeSessionId)
  const selectedStoredSessionIdRef = useRef<string | null>(selectedStoredSessionId)

  // Mirror the latest prop into its ref synchronously during render — not via
  // a passive useEffect, which only fires a frame after paint and left the
  // ref pointing at the outgoing session for one commit (#59305). Guarded to
  // fire only when the PROP itself changed since the last render (the same
  // condition a `useEffect(..., [activeSessionId])` dependency array already
  // enforced) rather than unconditionally: submit.ts and use-session-actions
  // pin these refs imperatively mid-flight (e.g. to a just-resumed runtime id)
  // without updating the source atom in lockstep, and wiring.tsx re-renders
  // constantly during an active turn — an unconditional resync would silently
  // clobber that pin on the next incidental render (#54527-class regression).
  const activeSessionIdPropRef = useRef(activeSessionId)

  if (activeSessionIdPropRef.current !== activeSessionId) {
    activeSessionIdPropRef.current = activeSessionId
    activeSessionIdRef.current = activeSessionId
  }

  const selectedStoredSessionIdPropRef = useRef(selectedStoredSessionId)

  if (selectedStoredSessionIdPropRef.current !== selectedStoredSessionId) {
    selectedStoredSessionIdPropRef.current = selectedStoredSessionId
    selectedStoredSessionIdRef.current = selectedStoredSessionId
  }

  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const sessionStateByRuntimeIdRef = useRef<SessionStateCache>(null!)

  if (sessionStateByRuntimeIdRef.current === null) {
    sessionStateByRuntimeIdRef.current = new SessionStateCache({
      isReferenced: (runtimeId, state) =>
        runtimeId === activeSessionIdRef.current ||
        state.storedSessionId === selectedStoredSessionIdRef.current ||
        $sessionTiles
          .get()
          .some(
            tile =>
              tile.runtimeId === runtimeId ||
              (state.storedSessionId !== null && tile.storedSessionId === state.storedSessionId)
          ),
      // A connection death mid-turn leaves snapshots whose frozen busy flags
      // will never settle (the respawned backend re-mints runtime ids), which
      // pinned megabytes of warm transcript per reconnect cycle behind
      // #isWarmSettled (#95189). Trust the cached in-flight flags only while
      // the authoritative store still claims work for the same runtime id.
      isAuthoritativelyActive: runtimeId => {
        const live = $sessionStates.get()[runtimeId]

        return Boolean(live && (live.busy || live.awaitingResponse))
      },
      onEvict: (runtimeId, state) => {
        // Ownership is removed with the transcript, but only if both sides still
        // describe this exact binding. A recycled runtime must not erase its
        // new owner's reverse entry.
        if (state.storedSessionId && runtimeIdByStoredSessionIdRef.current.get(state.storedSessionId) === runtimeId) {
          runtimeIdByStoredSessionIdRef.current.delete(state.storedSessionId)
        }

        releaseSessionTranscript(runtimeId)
      }
    })
  }

  const sessionStateCache = sessionStateByRuntimeIdRef.current
  const pendingViewStateRef = useRef<{ sessionId: string; state: ClientSessionState } | null>(null)
  const viewSyncRafRef = useRef<number | null>(null)
  const transcriptViewGateByRuntimeIdRef = useRef(new Map<string, symbol>())
  // Runtime id whose transcript currently occupies `$messages` — lets the
  // flush below tell a same-session refresh from a thread switch.
  const viewSessionIdRef = useRef<string | null>(null)

  // A runtime id is owned by the backend that minted it. Switching the active
  // connection (local ↔ remote gateway, or one remote to another) changes
  // which backend owns every stored session, so the cached stored→runtime
  // bindings from the PREVIOUS backend must not survive the switch: sending a
  // stale local runtime id to a remote gateway makes it reject the resume as
  // "Session not found" (#93888). Wipe both warm-cache refs on connection
  // change so the next resume resolves a fresh runtime from the new backend
  // (the cold `session.resume` path, which correctly uses the durable stored
  // id, repopulates them).
  const connection = useStore($connection)
  const connectionScopeKey = connection?.connectionId ?? connection?.mode ?? 'default'
  const prevConnectionScopeKeyRef = useRef(connectionScopeKey)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (prevConnectionScopeKeyRef.current === connectionScopeKey) {
      return
    }

    prevConnectionScopeKeyRef.current = connectionScopeKey
    runtimeIdByStoredSessionIdRef.current.clear()
    sessionStateCache.clear()
  }, [connectionScopeKey, sessionStateCache])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    setMutableRef(busyRef, busy)
  }, [busy, busyRef])

  const ensureSessionState = useCallback(
    (sessionId: string, storedSessionId?: string | null) => {
      const existing = sessionStateCache.get(sessionId)

      if (existing) {
        if (storedSessionId !== undefined && storedSessionId !== existing.storedSessionId) {
          // Stored id changed (e.g. auto-compression rotated it). Create a NEW
          // state object rather than mutating in place — updateSessionState needs
          // the PREVIOUS state to detect transitions (busy→idle, id rotation).
          const updated = invalidatePersistedDisplayTranscriptAuthority({ ...existing, storedSessionId })

          // Drop the obsolete stored→runtime reverse mapping as soon as the id
          // rotates (e.g. auto-compression forks a continuation). Leaving the
          // stale key lets getRuntimeIdForStoredSession resolve the old stored id
          // to this runtime, which the compression route-follow logic relies on
          // being absent. The rotation signal was previously emitted centrally
          // from handleTransition (session-states.ts), but updateSessionState
          // now skips publishSessionState (and thus handleTransition) when the
          // updater is a no-op — fire it here so the route-follow effect still
          // tracks compression without needing a dummy state write.
          if (existing.storedSessionId && existing.storedSessionId !== storedSessionId) {
            runtimeIdByStoredSessionIdRef.current.delete(existing.storedSessionId)

            // A rotation event needs a real next id — a null/cleared stored id
            // is a detach, not a rotation the route-follow effect should chase.
            if (storedSessionId && sessionId === $activeSessionId.get()) {
              setActiveSessionStoredIdRotation({
                nextStoredSessionId: storedSessionId,
                previousStoredSessionId: existing.storedSessionId,
                runtimeSessionId: sessionId
              })
            }
          }

          if (storedSessionId) {
            runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)
          }

          sessionStateCache.set(sessionId, updated)
        }

        return sessionStateCache.get(sessionId)!
      }

      const created = createClientSessionState(storedSessionId ?? null)

      if (storedSessionId) {
        runtimeIdByStoredSessionIdRef.current.set(storedSessionId, sessionId)
      }

      sessionStateCache.set(sessionId, created)

      return created
    },
    [sessionStateCache]
  )

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

  const holdSessionTranscriptView = useCallback((runtimeId: string): (() => void) => {
    const token = Symbol(runtimeId)
    transcriptViewGateByRuntimeIdRef.current.set(runtimeId, token)

    return () => {
      if (transcriptViewGateByRuntimeIdRef.current.get(runtimeId) === token) {
        transcriptViewGateByRuntimeIdRef.current.delete(runtimeId)
      }
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

      const viewState = suppressTranscriptForView(state, transcriptViewGateByRuntimeIdRef.current.has(sessionId))

      syncRuntimeMetadataToView(viewState)
      pendingViewStateRef.current = { sessionId, state: viewState }

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
      const isCriticalTransition = !viewState.busy || viewState.needsInput

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
      // Give the updater the raw previous state so it can return the same
      // reference when nothing changed (the caller sees a no-op). Previously
      // the param was always a fresh spread, so every call looked like a
      // change — including periodic ~1/s session.info heartbeats that churn
      // $sessionStates and its computed atoms on every tick.
      const next = updater(previous)

      // If the updater returned the same reference, nothing changed for this
      // session — skip the store write, publishSessionState, and view sync.
      // The cache entry was already updated by ensureSessionState (if
      // storedSessionId rotated); the caller gets its return value from the
      // cache, so stale reads don't regress.
      if (next === previous) {
        return previous
      }

      sessionStateCache.set(sessionId, next)
      // Crash-survivable turn progress: journal the running turn's visible
      // tail (throttled localStorage write; cleared the moment the turn
      // settles) so a renderer/app death mid-turn can be recovered on resume.
      persistInFlightTurnState(next)
      // Publishing to $sessionStates automatically fires transition side-effects
      // (watchdog, settle grace, unread marker, compression id rotation) inside
      // publishSessionState — no manual transition call needed.
      publishSessionState(sessionId, next)
      sessionStateCache.prune()
      syncSessionStateToView(sessionId, next)

      return next
    },
    [ensureSessionState, sessionStateCache, syncSessionStateToView]
  )

  useEffect(() => {
    sessionStateCache.prune()
  }, [activeSessionId, selectedStoredSessionId, sessionStateCache, sessionTiles])

  const getRuntimeIdForStoredSession = useCallback(
    (storedSessionId: string): string | null => {
      const runtimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)

      if (!runtimeId) {
        return null
      }

      const runtimeState = sessionStateCache.get(runtimeId)

      return runtimeState?.storedSessionId === storedSessionId ? runtimeId : null
    },
    [sessionStateCache]
  )

  return {
    activeSessionIdRef,
    ensureSessionState,
    getRuntimeIdForStoredSession,
    holdSessionTranscriptView,
    resetViewSync,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: sessionStateByRuntimeIdRef as MutableRefObject<Map<string, ClientSessionState>>,
    syncSessionStateToView,
    updateSessionState
  }
}
