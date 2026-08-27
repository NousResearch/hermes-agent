import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { preserveLocalAssistantErrors, sealOpenToolParts, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery, batteryPollInterval } from '@/store/power'
import { refreshActiveProfile } from '@/store/profile'
import {
  $activeSessionId,
  $busy,
  $currentCwd,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHint,
  sessionMatchesStoredId,
  setCurrentCwd
} from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import {
  $sessionStates,
  $sessionTiles,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS,
  setSessionStalled
} from '@/store/session-states'

import type { ClientSessionState } from '../../types'
import type { GatewayRequester } from '../types'

interface ActiveTranscriptSession {
  ownerRoute?: SessionProfileRoute
  profile?: string | null
}

/** Resolve an active transcript from visible rows or its unique hidden owner. */
export function resolveActiveTranscriptSession(storedSessionId: string): ActiveTranscriptSession | undefined {
  const visible =
    $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ??
    $messagingSessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

  if (visible) {
    return { profile: visible.profile }
  }

  const ownerRoute = getSessionOwnerHint(storedSessionId)

  return ownerRoute ? { ownerRoute, profile: ownerRoute.profile } : undefined
}

export interface ActiveTranscriptRefreshDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  requestSequenceRef: MutableRefObject<number>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  resolveSession: (storedSessionId: string) => ActiveTranscriptSession | null | undefined
  signatureRef: MutableRefObject<Map<string, string>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

function tileRuntimeOwnsLiveState(runtimeId: string): boolean {
  const state = $sessionStates.get()[runtimeId]

  return Boolean(state && (state.busy || state.awaitingResponse || state.needsInput || state.turnLive))
}

/**
 * Reconcile the persisted transcripts of every open WORKSPACE TILE (#93942
 * slice 1). Bot canonical chats live here — never in $sessions /
 * $messagingSessions (they carry the core `hidden` flag), so the main-pane
 * reconcile path's resolveSession() bails on them and a background delivery
 * never reaches an open bot chat. Each tile carries its own stored↔runtime id
 * pair, so no resolution step is needed; refreshes are signature-gated per
 * tile so a no-change event costs nothing, and a busy tile is skipped (its own
 * stream owns the view while streaming).
 *
 * Sequencing note (#94255 review): all tiles SHARE one request sequence, so a
 * second tick arriving mid-read invalidates every in-flight read from the
 * first (latest-wins — same discipline as the main pane path). Under rapid
 * tick bursts only the final tick lands updates; that is intended, since each
 * tick re-reads from storage anyway.
 */
export async function reconcileTileTranscripts({
  requestSequenceRef,
  signatureRef,
  updateSessionState,
  tiles: tilesOverride
}: {
  requestSequenceRef: MutableRefObject<number>
  signatureRef: MutableRefObject<Map<string, string>>
  tiles?: Array<{ storedSessionId: string; runtimeId?: string }>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}): Promise<void> {
  const tiles = tilesOverride ?? $sessionTiles.get()
  const openSignatureKeys = new Set(tiles.map(tile => `tile:${tile.storedSessionId}`))

  for (const signatureKey of signatureRef.current.keys()) {
    if (!openSignatureKeys.has(signatureKey)) {
      signatureRef.current.delete(signatureKey)
    }
  }

  for (const tile of tiles) {
    const storedSessionId = tile.storedSessionId
    const runtimeSessionId = tile.runtimeId

    if (!runtimeSessionId) {
      // Resume not yet bound — the tile's own stream owns the view.
      continue
    }

    if (!storedSessionId || !runtimeSessionId || tileRuntimeOwnsLiveState(runtimeSessionId)) {
      continue
    }

    if ($activeSessionId.get() === runtimeSessionId) {
      // The main pane reconcile already owns this surface.
      continue
    }

    const requestId = ++requestSequenceRef.current

    // With a tiles override (test path), the live $sessionTiles check can't
    // see the synthetic tile — treat override tiles as present.
    const tileStillPresent = () =>
      tilesOverride
        ? tilesOverride.some(t => t.storedSessionId === storedSessionId && t.runtimeId === runtimeSessionId)
        : $sessionTiles.get().some(t => t.storedSessionId === storedSessionId && t.runtimeId === runtimeSessionId)

    try {
      const latest = await getLatestSessionMessages(storedSessionId)

      if (requestId !== requestSequenceRef.current || tileRuntimeOwnsLiveState(runtimeSessionId) || !tileStillPresent()) {
        // Tile closed or superseded mid-read — discard AND prune its
        // signature so the map doesn't grow one entry per ever-opened tile
        // for the app's lifetime (#94255 review point 3).
        signatureRef.current.delete(`tile:${storedSessionId}`)

        continue
      }

      const signatureKey = `tile:${storedSessionId}`
      const signature = sessionMessagesSignature(latest.messages)

      if (signatureRef.current.get(signatureKey) === signature) {
        continue
      }

      signatureRef.current.set(signatureKey, signature)
      const messages = toChatMessages(latest.messages)

      updateSessionState(
        runtimeSessionId,
        state => ({
          ...state,
          messages: preserveLocalAssistantErrors(
            graftRefreshedTailOntoBackfill(messages, state.messages),
            state.messages
          )
        }),
        storedSessionId
      )
    } catch {
      // Non-fatal: the next change event retries.
    }
  }
}

/** Reconcile one persisted transcript snapshot into the currently viewed session. */
export async function reconcileActiveTranscript({
  activeSessionIdRef,
  busyRef,
  requestSequenceRef,
  resolveSession,
  selectedStoredSessionIdRef,
  signatureRef,
  updateSessionState
}: ActiveTranscriptRefreshDeps): Promise<void> {
  const storedSessionId = selectedStoredSessionIdRef.current
  const runtimeSessionId = activeSessionIdRef.current

  if (!storedSessionId || !runtimeSessionId || busyRef.current) {
    return
  }

  const stored = resolveSession(storedSessionId)

  if (!stored) {
    return
  }

  const requestId = requestSequenceRef.current + 1
  requestSequenceRef.current = requestId

  try {
    const profileScope: ProfileScope = stored.ownerRoute
      ? {
          connectionId: stored.ownerRoute.connectionId,
          profile: stored.ownerRoute.targetProfile ?? stored.ownerRoute.profile
        }
      : stored.profile

    const latest = await getLatestSessionMessages(storedSessionId, profileScope)

    if (
      requestId !== requestSequenceRef.current ||
      busyRef.current ||
      selectedStoredSessionIdRef.current !== storedSessionId ||
      activeSessionIdRef.current !== runtimeSessionId
    ) {
      return
    }

    const signatureKey = stored.ownerRoute
      ? JSON.stringify([
          stored.ownerRoute.connectionId,
          stored.ownerRoute.profile,
          stored.ownerRoute.targetProfile ?? '',
          stored.ownerRoute.mode ?? '',
          storedSessionId
        ])
      : `${stored.profile ?? 'default'}:${storedSessionId}`

    const signature = sessionMessagesSignature(latest.messages)

    if (signatureRef.current.get(signatureKey) === signature) {
      return
    }

    signatureRef.current.set(signatureKey, signature)
    const messages = toChatMessages(latest.messages)

    updateSessionState(
      runtimeSessionId,
      state => ({
        ...state,
        // The refresh re-reads only the newest tail page; graft it onto any
        // older pages "Show earlier" already backfilled instead of clobbering
        // them (see transcript-backfill).
        messages: preserveLocalAssistantErrors(graftRefreshedTailOntoBackfill(messages, state.messages), state.messages)
      }),
      storedSessionId
    )
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
// A typing burst keeps the composer's contentEditable input handling on the
// same renderer main thread as the list refresh above (#95033): with a large
// session store, one refresh pass can block keystroke echo long enough that
// input visibly stalls. While the keyboard is warm — any keydown in this
// renderer window, not just the composer — hold that pass and land it once
// shortly after the last keypress. Sidebar staleness during a burst is
// accepted; the lighter polls (active_list snapshot, cron, transcript
// backstops) keep their cadence because they carry liveness, not the heavy
// list reconciliation.
const TYPING_BURST_QUIET_MS = 1_500

interface LiveSessionStatusItem {
  id?: string
  last_active?: number
  session_key?: string
  status?: 'idle' | 'starting' | 'waiting' | 'working'
}

interface LiveSessionStatusResponse {
  sessions?: LiveSessionStatusItem[]
}

// Runtime ids this poll has seen live, per gateway profile. A profile only
// ever reaps what its OWN snapshot previously reported: background profiles are
// served by different gateways and never appear in this profile's active_list,
// so an unscoped reap would dark out every other profile's running rows.
const liveRuntimeIdsByProfile = new Map<string, Set<string>>()

// Renderer-wide keyboard warmth, tracked at module scope like the live-runtime
// bookkeeping above: any keydown anywhere in the window marks activity, and a
// burst stays warm for TYPING_BURST_QUIET_MS after the last key. IME
// composition still emits keydown (keyCode 229), so one listener covers both.
let lastRendererInputAt = 0

/** Record renderer-wide keyboard activity (wired to a capture-phase window
 *  keydown listener by useBackgroundSync). */
export function noteRendererKeyboardActivity(nowMs = Date.now()): void {
  lastRendererInputAt = nowMs
}

/** True while a typing burst is still warm enough to hold the heavy list
 *  refresh (see TYPING_BURST_QUIET_MS). */
export function isTypingBurstActive(nowMs = Date.now()): boolean {
  return nowMs - lastRendererInputAt < TYPING_BURST_QUIET_MS
}

function remainingTypingQuietMs(nowMs: number): number {
  return Math.max(0, TYPING_BURST_QUIET_MS - (nowMs - lastRendererInputAt))
}

/** Forget keyboard history — test isolation only (mirrors
 *  resetLiveRuntimeTracking). */
export function resetTypingActivityTracking(): void {
  lastRendererInputAt = 0
}

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
  const seen = new Set<string>()

  for (const session of response.sessions ?? []) {
    const runtimeSessionId = session.id?.trim()
    const storedSessionId = session.session_key?.trim()
    const needsInput = session.status === 'waiting'
    const working = session.status === 'working' || needsInput

    if (!runtimeSessionId || !storedSessionId) {
      continue
    }

    seen.add(runtimeSessionId)

    const existing = $sessionStates.get()[runtimeSessionId]

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
      publishSessionState(runtimeSessionId, {
        ...(existing ?? createClientSessionState(storedSessionId)),
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
  const previouslyLive = liveRuntimeIdsByProfile.get(profileKey)

  if (previouslyLive) {
    for (const runtimeSessionId of previouslyLive) {
      if (seen.has(runtimeSessionId)) {
        continue
      }

      const existing = $sessionStates.get()[runtimeSessionId]

      if (existing?.busy || existing?.needsInput || existing?.awaitingResponse) {
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

  liveRuntimeIdsByProfile.set(profileKey, seen)
}

/** Forget every profile's live-runtime bookkeeping. A gateway wipe already
 *  drops the session states these ids point at, so a carried-over set would
 *  only reap runtimes that no longer exist. */
export function resetLiveRuntimeTracking(): void {
  liveRuntimeIdsByProfile.clear()
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
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
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
  requestGateway,
  updateSessionState
}: BackgroundSyncParams): void {
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)
  const sessionsChangeTick = useStore($sessionsChangeTick)
  const activeTranscriptBusy = useStore($busy)
  const activeTranscriptRefreshPendingRef = useRef<string | null>(null)
  // Tile reconcile state (#93942 slice 1): shared sequence guard + per-tile
  // transcript signatures, so no-change ticks and closed tiles cost nothing.
  const tileRequestSequenceRef = useRef(0)
  const tileSignatureRef = useRef(new Map<string, string>())
  // Tile reconciliation reads each runtime's live state directly from
  // $sessionStates; the primary chat's $busy atom has no authority over tiles.

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
    let typingDeferTimer: null | number = null

    const run = () => {
      lastRunAt = Date.now()
      void refreshSessions()
      void refreshMessagingSessions()
      requestActiveTranscriptRefresh(true)
      // Bot canonical chats live in workspace tiles, never in the main-pane
      // selection — without this they never see background deliveries
      // (#93942 scenario A). Signature-gated per tile, so no-change ticks
      // cost nothing.
      void reconcileTileTranscripts({
        requestSequenceRef: tileRequestSequenceRef,
        signatureRef: tileSignatureRef,
        updateSessionState
      })
    }

    // Hold the coalesced pass while a typing burst is warm (#95033) so the
    // heavy list work never lands under keystrokes. One timer services every
    // caller: ticks that arrive mid-deferral find it already armed and return.
    // Fire time is the remaining quiet window, not a poll — a later key
    // extends lastRendererInputAt, and the firing callback re-arms if still
    // warm. There is no starvation cap: a continuous burst keeps holding.
    const runWhenKeyboardQuiet = () => {
      const now = Date.now()

      if (!isTypingBurstActive(now)) {
        if (typingDeferTimer !== null) {
          window.clearTimeout(typingDeferTimer)
          typingDeferTimer = null
        }

        run()

        return
      }

      if (typingDeferTimer === null) {
        typingDeferTimer = window.setTimeout(() => {
          typingDeferTimer = null
          runWhenKeyboardQuiet()
        }, remainingTypingQuietMs(now))
      }
    }

    const unsubscribe = $sessionsChangeTick.listen(() => {
      const since = Date.now() - lastRunAt

      if (since >= SESSIONS_LIST_TICK_GAP_MS) {
        runWhenKeyboardQuiet()
      } else if (typingDeferTimer === null && timer === null) {
        // Within the gap a pass is already scheduled — trailing timer or a
        // typing deferral. Arming another one here would stack extra passes.
        timer = window.setTimeout(() => {
          timer = null
          runWhenKeyboardQuiet()
        }, SESSIONS_LIST_TICK_GAP_MS - since)
      }
    })

    return () => {
      unsubscribe()

      if (timer !== null) {
        window.clearTimeout(timer)
      }

      if (typingDeferTimer !== null) {
        window.clearTimeout(typingDeferTimer)
      }
    }
  }, [
    changeEventsAvailable,
    gatewayState,
    refreshMessagingSessions,
    refreshSessions,
    requestActiveTranscriptRefresh,
    updateSessionState
  ])

  // Keyboard warmth for the deferral above: capture phase on window. Any
  // keydown in this renderer (composer, modal, settings) counts — conservative
  // on purpose. Pure timestamp write, no React state.
  useEffect(() => {
    const markInput = (): void => noteRendererKeyboardActivity()

    window.addEventListener('keydown', markInput, true)

    return () => {
      window.removeEventListener('keydown', markInput, true)
    }
  }, [])

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
