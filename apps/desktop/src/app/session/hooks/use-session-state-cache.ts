import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { preserveLocalAssistantErrors } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setMutableRef } from '@/lib/mutable-ref'
import { reconcilePetLiveSessionFocus, syncPetLiveSessionState } from '@/store/pet-live-session'
import { $activeGatewayProfile } from '@/store/profile'
import { normalizeProfileKey } from '@/store/profile-key'
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
import {
  deleteProfileSessionValue,
  getProfileSessionValue,
  profileSessionKey,
  setProfileSessionValue
} from '@/store/session-identity'

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
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const activeSessionIdRef = useRef<string | null>(null)
  const activeGatewayProfileRef = useRef(normalizeProfileKey(activeGatewayProfile))
  const selectedStoredSessionIdRef = useRef<string | null>(null)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const pendingViewStateRef = useRef<{ profile: string; sessionId: string; state: ClientSessionState } | null>(null)
  const viewSyncRafRef = useRef<number | null>(null)
  // Runtime id whose transcript currently occupies `$messages` — lets the
  // flush below tell a same-session refresh from a thread switch.
  const viewSessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId
  }, [activeSessionId])

  useEffect(() => {
    activeGatewayProfileRef.current = normalizeProfileKey(activeGatewayProfile)
  }, [activeGatewayProfile])

  useEffect(() => {
    setMutableRef(busyRef, busy)
  }, [busy, busyRef])

  useEffect(() => {
    selectedStoredSessionIdRef.current = selectedStoredSessionId
  }, [selectedStoredSessionId])

  useEffect(() => {
    const profile = normalizeProfileKey(activeGatewayProfile)

    const cachedState = activeSessionId
      ? getProfileSessionValue(sessionStateByRuntimeIdRef.current, profile, activeSessionId)
      : undefined

    reconcilePetLiveSessionFocus(
      activeSessionId ? { profile, runtimeSessionId: activeSessionId } : null,
      cachedState && activeSessionId
        ? {
            profile,
            runtimeSessionId: activeSessionId,
            storedSessionId: cachedState.storedSessionId,
            busy: cachedState.busy,
            needsInput: cachedState.needsInput,
            awaitingResponse: cachedState.awaitingResponse,
            turnStartedAt: cachedState.turnStartedAt
          }
        : null
    )
  }, [activeGatewayProfile, activeSessionId])

  const ensureSessionState = useCallback((
    sessionId: string,
    storedSessionId?: string | null,
    profile: string | null | undefined = activeGatewayProfileRef.current
  ) => {
    const profileKey = normalizeProfileKey(profile)
    const existing = getProfileSessionValue(sessionStateByRuntimeIdRef.current, profileKey, sessionId)

    if (existing) {
      if (storedSessionId !== undefined) {
        const previousStoredSessionId = existing.storedSessionId
        existing.storedSessionId = storedSessionId

        if (storedSessionId) {
          setProfileSessionValue(runtimeIdByStoredSessionIdRef.current, profileKey, storedSessionId, sessionId)

          if (existing.busy) {
            setSessionWorking(profileKey, storedSessionId, true)
          }
        }

        if (previousStoredSessionId && previousStoredSessionId !== storedSessionId) {
          deleteProfileSessionValue(runtimeIdByStoredSessionIdRef.current, profileKey, previousStoredSessionId)
          setSessionWorking(profileKey, previousStoredSessionId, false)
        }
      }

      return existing
    }

    const created = createClientSessionState(storedSessionId ?? null, [], profileKey)
    setProfileSessionValue(sessionStateByRuntimeIdRef.current, profileKey, sessionId, created)

    if (storedSessionId) {
      setProfileSessionValue(runtimeIdByStoredSessionIdRef.current, profileKey, storedSessionId, sessionId)
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

    if (
      !pending ||
      pending.sessionId !== activeSessionIdRef.current ||
      pending.profile !== normalizeProfileKey($activeGatewayProfile.get())
    ) {
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

    // On a thread switch `$messages` still holds the *previous* thread, so
    // preserving its local errors would graft that thread's failed turn (e.g.
    // an out-of-funds error) onto this one — then cascade it everywhere as the
    // polluted view becomes the next switch's baseline. Only carry errors
    // across a same-session refresh; our cached state already keeps its own.
    const nextMessages =
      viewSessionIdRef.current === profileSessionKey(pending.profile, pending.sessionId)
        ? preserveLocalAssistantErrors(pending.state.messages, currentMessages)
        : pending.state.messages

    if (!sameMessageList(nextMessages, currentMessages)) {
      setMessages(nextMessages)
    }

    viewSessionIdRef.current = profileSessionKey(pending.profile, pending.sessionId)

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
    (
      sessionId: string,
      state: ClientSessionState,
      profile: string | null | undefined = state.profile || activeGatewayProfileRef.current
    ) => {
      const profileKey = normalizeProfileKey(profile)

      // Only the currently-viewed session may stage into the shared `$messages`
      // view. A background session (e.g. one still busy and emitting stream /
      // error updates after the user toggled away) must update its own cache
      // entry but never the view — otherwise its messages clobber the
      // foreground transcript and appear to "bleed" into every other session.
      // The flush below also re-checks the active id, but staging here is what
      // prevents a background write from overwriting an already-pending
      // foreground write within the same animation frame (only one RAF is
      // scheduled, so the last `pendingViewStateRef` writer would otherwise win).
      if (sessionId !== activeSessionIdRef.current || profileKey !== normalizeProfileKey($activeGatewayProfile.get())) {
        return
      }

      syncRuntimeMetadataToView(state)
      pendingViewStateRef.current = { profile: profileKey, sessionId, state }

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
      storedSessionId?: string | null,
      profile: string | null | undefined = activeGatewayProfileRef.current
    ) => {
      const profileKey = normalizeProfileKey(profile)
      const previous = ensureSessionState(sessionId, storedSessionId, profileKey)
      const next = { ...updater({ ...previous, messages: previous.messages }), profile: profileKey }
      setProfileSessionValue(sessionStateByRuntimeIdRef.current, profileKey, sessionId, next)

      syncPetLiveSessionState(
        {
          profile: profileKey,
          runtimeSessionId: sessionId,
          storedSessionId: next.storedSessionId,
          busy: next.busy,
          needsInput: next.needsInput,
          awaitingResponse: next.awaitingResponse,
          turnStartedAt: next.turnStartedAt
        },
        activeSessionIdRef.current
          ? { profile: activeGatewayProfileRef.current, runtimeSessionId: activeSessionIdRef.current }
          : null
      )

      if (previous.storedSessionId !== next.storedSessionId || !next.busy) {
        setSessionWorking(profileKey, previous.storedSessionId, false)
      }

      if (previous.storedSessionId !== next.storedSessionId || !next.needsInput) {
        setSessionAttention(profileKey, previous.storedSessionId, false)
      }

      setSessionWorking(profileKey, next.storedSessionId, next.busy)
      setSessionAttention(profileKey, next.storedSessionId, next.needsInput)

      // Every state update is effectively a "still alive" heartbeat for
      // streaming events. The session-store watchdog uses this to keep the
      // working flag alive during long-running turns and to clear it once
      // the stream goes silent.
      if (next.busy) {
        noteSessionActivity(profileKey, next.storedSessionId)
      }

      syncSessionStateToView(sessionId, next, profileKey)

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
      onSessionWatchdogClear(identity => {
        const runtimeId = getProfileSessionValue(
          runtimeIdByStoredSessionIdRef.current,
          identity.profile,
          identity.sessionId
        )

        const state = runtimeId
          ? getProfileSessionValue(sessionStateByRuntimeIdRef.current, identity.profile, runtimeId)
          : undefined

        if (!runtimeId || !state?.busy) {
          return
        }

        updateSessionState(
          runtimeId,
          current => ({
            ...current,
            awaitingResponse: false,
            busy: false,
            needsInput: false
          }),
          identity.sessionId,
          identity.profile
        )
      }),
    [updateSessionState]
  )

  return {
    activeSessionIdRef,
    ensureSessionState,
    resetViewSync,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  }
}
