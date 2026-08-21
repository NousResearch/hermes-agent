import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages } from '@/hermes'
import { preserveLocalAssistantErrors, sealOpenToolParts, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery, batteryPollInterval } from '@/store/power'
import { normalizeProfileKey, refreshActiveProfile } from '@/store/profile'
import {
  $activeSessionId,
  $busy,
  $currentCwd,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setCurrentCwd
} from '@/store/session'
import {
  $sessionStates,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS,
  setSessionStalled
} from '@/store/session-states'

import { sessionStateMatchesOwner, type SessionStateOwner } from '../../session/session-state-cache'
import type { ClientSessionState } from '../../types'
import type { GatewayRequester } from '../types'

interface ActiveTranscriptSession {
  profile?: string | null
}

/** Resolve an active transcript from either local recents or messaging slices. */
export function resolveActiveTranscriptSession(
  storedSessionId: string,
  ownerProfile: string
): ActiveTranscriptSession | undefined {
  const owner = normalizeProfileKey(ownerProfile)
  const matchesOwner = (session: ActiveTranscriptSession) => normalizeProfileKey(session.profile) === owner

  return (
    $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId) && matchesOwner(session)) ??
    $messagingSessions.get().find(session => sessionMatchesStoredId(session, storedSessionId) && matchesOwner(session))
  )
}

export interface ActiveTranscriptRefreshDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  requestSequenceRef: MutableRefObject<number>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionProfileRef: MutableRefObject<string | null>
  resolveSession: (storedSessionId: string, ownerProfile: string) => ActiveTranscriptSession | null | undefined
  sessionStateHasOwner: (sessionId: string, owner: SessionStateOwner) => boolean
  signatureRef: MutableRefObject<Map<string, string>>
  updateOwnedSessionState: (
    sessionId: string,
    owner: SessionStateOwner,
    updater: (state: ClientSessionState) => ClientSessionState
  ) => boolean
}

/** Reconcile one persisted transcript snapshot into the currently viewed session. */
export async function reconcileActiveTranscript({
  activeSessionIdRef,
  busyRef,
  requestSequenceRef,
  resolveSession,
  selectedStoredSessionIdRef,
  selectedStoredSessionProfileRef,
  sessionStateHasOwner,
  signatureRef,
  updateOwnedSessionState
}: ActiveTranscriptRefreshDeps): Promise<void> {
  const storedSessionId = selectedStoredSessionIdRef.current
  const storedSessionProfile = selectedStoredSessionProfileRef.current
  const runtimeSessionId = activeSessionIdRef.current

  if (!storedSessionId || !storedSessionProfile || !runtimeSessionId || busyRef.current) {
    return
  }

  const stored = resolveSession(storedSessionId, storedSessionProfile)

  if (!stored) {
    return
  }

  const requestId = requestSequenceRef.current + 1
  requestSequenceRef.current = requestId

  try {
    const latest = await getLatestSessionMessages(storedSessionId, storedSessionProfile)

    if (
      requestId !== requestSequenceRef.current ||
      busyRef.current ||
      selectedStoredSessionIdRef.current !== storedSessionId ||
      selectedStoredSessionProfileRef.current !== storedSessionProfile ||
      activeSessionIdRef.current !== runtimeSessionId
    ) {
      return
    }

    const signatureKey = `${storedSessionProfile}:${storedSessionId}`
    const signature = sessionMessagesSignature(latest.messages)

    if (signatureRef.current.get(signatureKey) === signature) {
      return
    }

    const messages = toChatMessages(latest.messages)
    const owner = { profile: storedSessionProfile, storedSessionId }

    if (
      !updateOwnedSessionState(runtimeSessionId, owner, state => ({
        ...state,
        // The refresh re-reads only the newest tail page; graft it onto any
        // older pages "Show earlier" already backfilled instead of clobbering
        // them (see transcript-backfill).
        messages: preserveLocalAssistantErrors(graftRefreshedTailOntoBackfill(messages, state.messages), state.messages)
      })) ||
      !sessionStateHasOwner(runtimeSessionId, owner)
    ) {
      return
    }

    // Only suppress a future refresh after this snapshot was actually accepted
    // by the captured composite cache owner.
    signatureRef.current.set(signatureKey, signature)
  } catch {
    // Non-fatal: the next change event or manual resume can hydrate the view.
  }
}

// Cron sessions are written by a background scheduler tick, messaging turns by
// the background gateway (Telegram, WeChat, Discord, …) — neither signals the
// desktop websocket directly. Backends with the change watcher broadcast
// `cron.changed` / `sessions.changed` when those on-disk writes land, so the
// timers below become slow safety-net backstops; against an older backend
// (no `change_events` on gateway.ready) they stay at the legacy cadence.
const CRON_POLL_INTERVAL_MS = 30_000
const CRON_BACKSTOP_INTERVAL_MS = 5 * 60_000
const MESSAGING_POLL_INTERVAL_MS = 10_000
const ACTIVE_MESSAGING_SESSION_POLL_INTERVAL_MS = 5_000
const ACTIVE_MESSAGING_SESSION_BACKSTOP_INTERVAL_MS = 30_000
// Match the TUI's live-session refresh cadence. Auto-compression can rotate a
// stored session id while its turn keeps running; until the next snapshot the
// sidebar row points at the new id while the renderer still knows the old one.
// A 15s cadence made that healthy transition look finished long enough to be
// alarming (and clicking the row appeared to "fix" it by touching the live
// session). This snapshot is small and already polled at 1.5s by the TUI.
const LIVE_SESSION_STATUS_POLL_INTERVAL_MS = 1_500
// With change events the snapshot re-pulls on every sessions.changed tick, so
// the interval only covers the degraded-socket edge the stream can't replay
// (see rehydrateLiveSessionStatuses) — 30s is plenty for that.
const LIVE_SESSION_STATUS_BACKSTOP_INTERVAL_MS = 30_000
// Coalesce tick-driven sidebar list refreshes: sessions.changed fires (floored
// to 2s server-side) on every state.db write during a streaming turn, and the
// full list refresh is heavier than the active_list snapshot. Trailing-edge
// scheduled, so the burst's last write always lands.
const SESSIONS_LIST_TICK_GAP_MS = 10_000

interface LiveSessionStatusItem {
  id?: string
  last_active?: number
  session_key?: string
  status?: 'idle' | 'starting' | 'waiting' | 'working'
}

interface LiveSessionStatusResponse {
  sessions?: LiveSessionStatusItem[]
}

// Runtime → stored-session ownership this poll has seen live, per gateway
// profile. Keeping both ids is load-bearing: a runtime id can be reused between
// snapshots, including by another profile, and the old snapshot must not reap
// or rewrite the replacement (ABA).
const liveRuntimeOwnersByProfile = new Map<string, Map<string, string>>()

/** Restore sidebar liveness after a renderer/backend reconnect. Stream events
 * normally own these states, but events emitted while Desktop was disconnected
 * cannot be replayed. `session.active_list` is the authoritative in-memory
 * snapshot and does not resume, focus, or otherwise mutate a chat.
 *
 * The snapshot is authoritative about ABSENCE too. A turn that ends while the
 * websocket is degraded — a remote gateway over a flaky link, a reconnect, a
 * profile swap — drops out of `_sessions` without Desktop ever seeing the
 * `running: false` edge, so the row keeps spinning and the busy→idle transition
 * that paints the green "your turn" dot never fires. Reaping runtimes that
 * vanish between polls restores both. */
export function rehydrateLiveSessionStatuses(
  response: LiveSessionStatusResponse,
  nowMs = Date.now(),
  profileKey = 'default'
): void {
  const ownerProfile = normalizeProfileKey(profileKey)
  const occupiedRuntimeIds = new Set<string>()
  const seen = new Map<string, string>()

  for (const session of response.sessions ?? []) {
    const runtimeSessionId = session.id?.trim()
    const storedSessionId = session.session_key?.trim()
    const needsInput = session.status === 'waiting'
    const working = session.status === 'working' || needsInput

    if (!runtimeSessionId || !storedSessionId) {
      continue
    }

    // Even when the durable owner conflicts with the renderer cache, this
    // profile's current snapshot proves the runtime slot is occupied. Record
    // that fact before the fail-closed owner check so the absence reaper cannot
    // misclassify the rejected replacement as disappearance of the old owner.
    occupiedRuntimeIds.add(runtimeSessionId)

    const existing = $sessionStates.get()[runtimeSessionId]
    const owner = { profile: ownerProfile, storedSessionId }

    // Runtime ids are transport-local and can collide or be reused. Once a
    // slot exists, this snapshot may touch it only when BOTH durable owner
    // coordinates still match. Unknown profile provenance also fails closed.
    if (existing && !sessionStateMatchesOwner(existing, owner)) {
      continue
    }

    seen.set(runtimeSessionId, storedSessionId)

    // A turn we just submitted is not yet running as far as the backend is
    // concerned, so the snapshot honestly reports it idle — but the local
    // stream is already waiting on its first token, and it is the newer
    // information. The stream path refuses to clear busy in exactly this window
    // (`awaitingResponse && !sawAssistantPayload`); without the same refusal
    // here a poll lands between submit and first token and darkens the row.
    const busy = working || Boolean(existing?.awaitingResponse && !existing.sawAssistantPayload)

    // Avoid re-arming the watchdog on every poll. Publish only when the
    // authoritative live snapshot differs from the renderer mirror; normal
    // gateway events continue to own subsequent transitions.
    if (
      !existing ||
      existing.storedSessionId !== storedSessionId ||
      existing.busy !== busy ||
      existing.needsInput !== needsInput
    ) {
      const ownedState = existing ?? { ...createClientSessionState(storedSessionId), profile: ownerProfile }

      publishSessionState(runtimeSessionId, {
        ...ownedState,
        busy,
        needsInput,
        storedSessionId
      })
    }

    if (!working) {
      setSessionStalled(storedSessionId, false)

      continue
    }

    const lastActiveMs = Number(session.last_active) * 1000

    const isQuiet =
      session.status === 'working' &&
      Number.isFinite(lastActiveMs) &&
      lastActiveMs > 0 &&
      nowMs - lastActiveMs >= SESSION_WATCHDOG_TIMEOUT_MS

    setSessionStalled(storedSessionId, isQuiet)
  }

  // A runtime this profile's snapshot reported live LAST poll but not this one
  // has ended: the gateway reaps a session out of `_sessions` when its turn
  // completes and its transport goes away. Settle it through the normal publish
  // path so the busy→idle transition fires — that edge is what clears the
  // spinner AND marks the row unread ("your turn"). Only ids this profile
  // previously saw are eligible, so another profile's live rows are untouched.
  const previouslyLive = liveRuntimeOwnersByProfile.get(ownerProfile)

  if (previouslyLive) {
    for (const [runtimeSessionId, storedSessionId] of previouslyLive) {
      if (occupiedRuntimeIds.has(runtimeSessionId)) {
        continue
      }

      const existing = $sessionStates.get()[runtimeSessionId]
      const owner = { profile: ownerProfile, storedSessionId }

      if (
        sessionStateMatchesOwner(existing, owner) &&
        (existing.busy || existing.needsInput || existing.awaitingResponse)
      ) {
        publishSessionState(runtimeSessionId, {
          ...existing,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          streamId: null,
          turnStartedAt: null,
          turnLive: false,
          // The turn ended without its completion events reaching us — a lost
          // `tool.complete` would otherwise leave a spinning tool row in an
          // idle session. Seal open tool parts the same way the settle path
          // does, so the transcript matches the state.
          messages: sealOpenToolParts(existing.messages)
        })
      }
    }
  }

  liveRuntimeOwnersByProfile.set(ownerProfile, seen)
}

/** Forget every profile's live-runtime bookkeeping. A gateway wipe already
 *  drops the session states these ids point at, so a carried-over set would
 *  only reap runtimes that no longer exist. */
export function resetLiveRuntimeTracking(): void {
  liveRuntimeOwnersByProfile.clear()
}

interface BackgroundSyncParams {
  activeConnectionId: null | string
  activeGatewayProfile: string
  activeIsMessaging: boolean
  activeSessionId: null | string
  activeStoredSessionId: null | string
  freshDraftReady: boolean
  gatewayState: string
  refreshActiveTranscript: () => Promise<unknown> | unknown
  refreshCronJobs: () => Promise<unknown> | unknown
  refreshCurrentModel: (force?: boolean) => Promise<unknown> | unknown
  refreshHermesConfig: () => Promise<unknown> | unknown
  refreshMessagingSessions: () => Promise<unknown> | unknown
  refreshSessions: () => Promise<unknown> | unknown
  requestGateway: GatewayRequester
}

/** Poll a callback while the tab is visible, on `intervalMs`; re-checks on tab
 *  re-focus. On battery the cadence stretches (see store/power) — these are
 *  safety-net refreshes, not the live path, so they're the right thing to slow
 *  when the machine is spending its charge. Returns nothing — meant to live
 *  inside an effect. */
export function windowIsActivelyViewed({
  focused,
  visibilityState
}: {
  focused: boolean
  visibilityState: DocumentVisibilityState
}): boolean {
  return visibilityState === 'visible' && focused
}

function visiblePoll(intervalMs: number, tick: () => void): () => void {
  const run = () => {
    // On macOS an unfocused or app-hidden BrowserWindow commonly remains
    // `visibilityState === "visible"`. Visibility alone therefore kept every
    // safety-net gateway poll alive while the user was in another app. These
    // are stale-data backstops, not the live event path, so pause them until
    // the window is actually being viewed and catch up immediately on focus.
    if (windowIsActivelyViewed({ focused: document.hasFocus(), visibilityState: document.visibilityState })) {
      tick()
    }
  }

  let intervalId = window.setInterval(run, batteryPollInterval(intervalMs, $onBattery.get()))

  const unsubscribeBattery = $onBattery.listen(onBattery => {
    window.clearInterval(intervalId)
    intervalId = window.setInterval(run, batteryPollInterval(intervalMs, onBattery))
  })

  document.addEventListener('visibilitychange', run)
  window.addEventListener('focus', run)

  return () => {
    unsubscribeBattery()
    window.clearInterval(intervalId)
    document.removeEventListener('visibilitychange', run)
    window.removeEventListener('focus', run)
  }
}

/**
 * Keeps app data live while the gateway is open: an on-connect reseed (model /
 * profile / sessions + relative-cwd resolution), the cron / messaging /
 * open-transcript visibility polls, and the fresh-draft model/config reseed.
 * All the "the desktop websocket won't tell us, so poll" logic in one place.
 */
export function useBackgroundSync({
  activeConnectionId,
  activeGatewayProfile,
  activeIsMessaging,
  activeSessionId,
  activeStoredSessionId,
  freshDraftReady,
  gatewayState,
  refreshActiveTranscript,
  refreshCronJobs,
  refreshCurrentModel,
  refreshHermesConfig,
  refreshMessagingSessions,
  refreshSessions,
  requestGateway
}: BackgroundSyncParams): void {
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)
  const sessionsChangeTick = useStore($sessionsChangeTick)
  const activeTranscriptBusy = useStore($busy)
  const activeTranscriptRefreshPendingRef = useRef<string | null>(null)

  const requestActiveTranscriptRefresh = useCallback(
    (preservePending: boolean) => {
      if (!activeStoredSessionId || !activeSessionId) {
        return
      }

      const storedSessionId = activeStoredSessionId
      const runtimeSessionId = activeSessionId
      const sessionKey = `${storedSessionId}:${runtimeSessionId}`

      if (preservePending) {
        activeTranscriptRefreshPendingRef.current = sessionKey
      }

      if ($busy.get()) {
        return
      }

      if (preservePending && activeTranscriptRefreshPendingRef.current === sessionKey) {
        activeTranscriptRefreshPendingRef.current = null
      }

      let sawBusyDuringRead = false

      const unsubscribeBusy = $busy.listen(busy => {
        sawBusyDuringRead ||= busy
      })

      void Promise.resolve(refreshActiveTranscript()).finally(() => {
        unsubscribeBusy()

        // If streaming began while the read was in flight, reconciliation was
        // discarded and the external event still needs one idle retry.
        if (
          preservePending &&
          (sawBusyDuringRead || $busy.get()) &&
          $activeSessionId.get() === runtimeSessionId &&
          $selectedStoredSessionId.get() === storedSessionId
        ) {
          activeTranscriptRefreshPendingRef.current = sessionKey
        }
      })
    },
    [activeSessionId, activeStoredSessionId, refreshActiveTranscript]
  )

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void refreshCurrentModel()
    void refreshActiveProfile()
    void refreshSessions()

    // A RELATIVE workspace cwd (config `terminal.cwd: .`) renders as "." in the
    // file tree header — resolve it to the backend's absolute path once.
    // Session runtime info still overrides later, and never while a session is
    // active.
    const cwd = $currentCwd.get().trim()

    if (!$activeSessionId.get() && cwd && !/^(\/|[A-Za-z]:[\\/])/.test(cwd)) {
      void requestGateway<{ cwd?: string }>('config.get', { key: 'project', cwd })
        .then(info => {
          if (info.cwd && !$activeSessionId.get()) {
            setCurrentCwd(info.cwd)
          }
        })
        .catch(() => undefined)
    }
  }, [activeConnectionId, activeGatewayProfile, gatewayState, refreshCurrentModel, refreshSessions, requestGateway])

  // A reconnect loses renderer-only working/attention atoms while the backend
  // keeps the actual turns alive. Re-seed from the gateway's in-memory session
  // registry immediately, then re-pull on every sessions.changed broadcast; a
  // slow visible poll remains as the backstop for the degraded-socket edge the
  // stream cannot replay (legacy cadence against older backends).
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    let cancelled = false
    let inFlight = false

    const refreshLiveStatuses = async () => {
      if (inFlight) {
        return
      }

      inFlight = true

      try {
        const response = await requestGateway<LiveSessionStatusResponse>('session.active_list', {})

        if (!cancelled) {
          rehydrateLiveSessionStatuses(response, Date.now(), activeGatewayProfile)
        }
      } catch {
        // Older gateways may not expose session.active_list. Live stream events
        // still work as before; leave the current sidebar state untouched.
      } finally {
        inFlight = false
      }
    }

    const dispose = visiblePoll(
      changeEventsAvailable ? LIVE_SESSION_STATUS_BACKSTOP_INTERVAL_MS : LIVE_SESSION_STATUS_POLL_INTERVAL_MS,
      () => void refreshLiveStatuses()
    )

    void refreshLiveStatuses()

    return () => {
      cancelled = true
      dispose()
    }
    // sessionsChangeTick: each sessions.changed broadcast re-seeds immediately
    // via the effect re-run (already coalesced to 2s server-side).
  }, [activeGatewayProfile, changeEventsAvailable, gatewayState, requestGateway, sessionsChangeTick])

  // sessions.changed also means the *stored* list may have new rows (a cron
  // run's session, an inbound messaging turn creating a thread). The full list
  // refresh is heavier than the active_list snapshot, so trail it on a gap
  // instead of firing per tick. Direct atom subscription: the throttle state
  // lives in the effect closure, not in refs synced from renders.
  useEffect(() => {
    if (gatewayState !== 'open' || !changeEventsAvailable) {
      return
    }

    let lastRunAt = 0
    let timer: null | number = null

    const run = () => {
      lastRunAt = Date.now()
      void refreshSessions()
      void refreshMessagingSessions()
      requestActiveTranscriptRefresh(true)
    }

    const unsubscribe = $sessionsChangeTick.listen(() => {
      const since = Date.now() - lastRunAt

      if (since >= SESSIONS_LIST_TICK_GAP_MS) {
        run()
      } else if (timer === null) {
        timer = window.setTimeout(() => {
          timer = null
          run()
        }, SESSIONS_LIST_TICK_GAP_MS - since)
      }
    })

    return () => {
      unsubscribe()

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [changeEventsAvailable, gatewayState, refreshMessagingSessions, refreshSessions, requestActiveTranscriptRefresh])

  // Keep the cron-jobs section live without a user action (scheduler ticks in
  // the background). cron.changed (jobs.json moved: CRUD or a scheduler tick's
  // bookkeeping) drives the refresh; the visible poll is the backstop.
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    if (cronChangeTick > 0) {
      void refreshCronJobs()
    }

    return visiblePoll(
      changeEventsAvailable ? CRON_BACKSTOP_INTERVAL_MS : CRON_POLL_INTERVAL_MS,
      () => void refreshCronJobs()
    )
  }, [changeEventsAvailable, cronChangeTick, gatewayState, refreshCronJobs])

  // A busy transition only consumes a pending sessions.changed refresh. It
  // never creates one, so an ordinary local turn going busy -> idle does not
  // add a REST read. The event itself is coalesced by the list throttle above.
  useEffect(() => {
    if (
      gatewayState !== 'open' ||
      activeTranscriptBusy ||
      !activeSessionId ||
      !activeStoredSessionId ||
      activeTranscriptRefreshPendingRef.current !== `${activeStoredSessionId}:${activeSessionId}`
    ) {
      return
    }

    requestActiveTranscriptRefresh(true)
  }, [activeSessionId, activeStoredSessionId, activeTranscriptBusy, gatewayState, requestActiveTranscriptRefresh])

  // Preserve the pre-existing messaging behavior: refresh once when a
  // messaging transcript opens, then keep its visibility backstop. Desktop
  // sessions never enter this effect and therefore gain no periodic timer.
  useEffect(() => {
    if (gatewayState !== 'open' || !activeIsMessaging || !activeSessionId || !activeStoredSessionId) {
      return
    }

    const runScheduledRefresh = () => requestActiveTranscriptRefresh(false)

    runScheduledRefresh()

    return visiblePoll(
      changeEventsAvailable ? ACTIVE_MESSAGING_SESSION_BACKSTOP_INTERVAL_MS : ACTIVE_MESSAGING_SESSION_POLL_INTERVAL_MS,
      runScheduledRefresh
    )
  }, [
    activeIsMessaging,
    activeSessionId,
    activeStoredSessionId,
    changeEventsAvailable,
    gatewayState,
    requestActiveTranscriptRefresh
  ])

  // Messaging session lists against an older backend: no sessions.changed, so
  // keep the legacy visible poll. (Event-capable backends fold this into the
  // trailing sessions.changed refresh above.)
  useEffect(() => {
    if (gatewayState !== 'open' || changeEventsAvailable) {
      return
    }

    return visiblePoll(MESSAGING_POLL_INTERVAL_MS, () => void refreshMessagingSessions())
  }, [changeEventsAvailable, gatewayState, refreshMessagingSessions])

  // A fresh new-session draft (gateway open, no active session) re-pulls the
  // model + config so the composer pill reflects the profile default.
  useEffect(() => {
    if (gatewayState === 'open' && !activeSessionId && freshDraftReady) {
      void refreshCurrentModel()
      void refreshHermesConfig()
    }
  }, [activeSessionId, freshDraftReady, gatewayState, refreshCurrentModel, refreshHermesConfig])
}
