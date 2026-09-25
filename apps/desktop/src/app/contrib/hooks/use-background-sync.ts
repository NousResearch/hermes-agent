import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, preserveLocalAssistantErrors, sealOpenToolParts, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import { latestSessionTodos } from '@/lib/todos'
import { pendingSessionReplay } from '@/store/gateway'
import { $sidebarShowArchived } from '@/store/layout'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery, batteryPollInterval } from '@/store/power'
import { refreshActiveProfile } from '@/store/profile'
import { refreshProjectTree } from '@/store/projects'
import {
  $activeSessionId,
  $busy,
  $currentCwd,
  $selectedStoredSessionId,
  getSessionOwnerHint,
  ownerLookupSessionRows,
  sessionMatchesStoredId,
  setCurrentCwd
} from '@/store/session'
import type { SessionOwnerRoute } from '@/store/session-request-router'
import {
  $sessionStates,
  $sessionTiles,
  confirmReconnectSettlesExcept,
  noteSessionEvent,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS,
  setSessionStalled
} from '@/store/session-states'
import { loadArchivedSessions } from '@/store/sidebar-archive'
import { clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'

import type { ClientSessionState } from '../../types'
import type { GatewayRequester } from '../types'

interface ActiveTranscriptSession {
  ownerRoute?: SessionOwnerRoute
  profile?: string | null
}

/** Profile/connection scope used to read an active transcript from storage. */
export function profileScopeForTranscriptSession(stored: ActiveTranscriptSession | undefined): ProfileScope {
  if (!stored) {
    return undefined
  }

  if (stored.ownerRoute) {
    return {
      connectionId: stored.ownerRoute.connectionId,
      profile: stored.ownerRoute.targetProfile ?? stored.ownerRoute.profile
    }
  }

  return stored.profile
}

/** Only a runtime-matched explicit tile owner overrides visible rows; unique hints are the last fallback. */
export function resolveActiveTranscriptSession(
  storedSessionId: string,
  runtimeSessionId: string
): ActiveTranscriptSession | undefined {
  const verifiedOwner = $sessionTiles
    .get()
    .find(tile => tile.storedSessionId === storedSessionId && tile.runtimeId === runtimeSessionId)?.ownerRoute

  if (verifiedOwner) {
    return { ownerRoute: verifiedOwner, profile: verifiedOwner.profile }
  }

  const visible = ownerLookupSessionRows().find(session => sessionMatchesStoredId(session, storedSessionId))

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
  resolveSession: (storedSessionId: string, runtimeSessionId: string) => ActiveTranscriptSession | null | undefined
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

/** Backfill/retention may prepend or release a prefix without changing the tail. */
function transcriptChangedDuringRead(before: ChatMessage[] | undefined, after: ChatMessage[] | undefined): boolean {
  if (before === after) {
    return false
  }

  if (!before?.length || !after?.length) {
    return true
  }

  // Require the entire shorter transcript to be an unchanged suffix. Matching
  // only the last row would miss edits/tool updates earlier in the current turn.
  const overlap = Math.min(before.length, after.length)

  for (let offset = 1; offset <= overlap; offset += 1) {
    if (before[before.length - offset] !== after[after.length - offset]) {
      return true
    }
  }

  return false
}

/** Zero persisted rows read over a populated runtime bound to the SAME stored
 *  session. An empty page is not proof the transcript is empty — it is also
 *  what a respawning backend (or a state.db read racing the change event)
 *  returns. A runtime rebound to another stored id while the read was in
 *  flight holds no evidence about the requested transcript. */
function emptyPageOverPopulatedTranscript(
  page: readonly unknown[],
  current: ClientSessionState | undefined,
  storedSessionId: string
): boolean {
  return page.length === 0 && Boolean(current?.messages.length) && current?.storedSessionId === storedSessionId
}

type TileTranscriptTarget = { ownerRoute?: SessionOwnerRoute; storedSessionId: string; runtimeId?: string }

/** Signature key per tile — carries the owner route so two connections/profiles
 *  sharing a stored id (or a tile re-homed to another owner) never alias. */
function tileTranscriptSignatureKey(tile: TileTranscriptTarget): string {
  const route = tile.ownerRoute

  return `tile:${route ? `${route.connectionId}:${route.targetProfile ?? route.profile}:` : ''}${tile.storedSessionId}`
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
  tiles?: TileTranscriptTarget[]
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}): Promise<void> {
  const tiles = tilesOverride ?? $sessionTiles.get()
  const openSignatureKeys = new Set(tiles.map(tileTranscriptSignatureKey))

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

    // Bot tiles are pinned to an exact owner (connection + target profile);
    // read from that backend, not whichever profile is foreground. Tiles
    // without a route keep the legacy local read.
    const profileScope = profileScopeForTranscriptSession(tile)

    const signatureKey = tileTranscriptSignatureKey(tile)

    try {
      const replay = pendingSessionReplay(runtimeSessionId)

      if (replay && !(await replay)) {
        continue
      }

      if (
        requestId !== requestSequenceRef.current ||
        !tileStillPresent() ||
        tileRuntimeOwnsLiveState(runtimeSessionId)
      ) {
        continue
      }

      const messagesAtRequest = $sessionStates.get()[runtimeSessionId]?.messages
      // Passive: a hidden tile's refresh must never cold-start its owner
      // backend or hold a pool slot (#103375); no warm backend = retry next tick.
      const latest = await getLatestSessionMessages(storedSessionId, profileScope, { passive: true })
      const replayAtReturn = pendingSessionReplay(runtimeSessionId)

      if (replayAtReturn && !(await replayAtReturn)) {
        continue
      }

      const current = $sessionStates.get()[runtimeSessionId]

      if (
        requestId !== requestSequenceRef.current ||
        tileRuntimeOwnsLiveState(runtimeSessionId) ||
        transcriptChangedDuringRead(messagesAtRequest, current?.messages) ||
        !tileStillPresent()
      ) {
        // Tile closed or superseded mid-read — discard AND prune its
        // signature so the map doesn't grow one entry per ever-opened tile
        // for the app's lifetime (#94255 review point 3).
        signatureRef.current.delete(signatureKey)

        continue
      }

      // Same rule as the active pane below: a transient zero-row page must not
      // blank a populated tile, and leaves no signature behind.
      if (emptyPageOverPopulatedTranscript(latest.messages, current, storedSessionId)) {
        continue
      }

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
          // An un-acked optimistic `user-*` row exists only here, so a
          // background refresh that lands mid-send would drop it and the
          // message would have to be retyped. Same composition order as
          // reconcileAuthoritativeChatMessages (use-session-actions/index.ts).
          messages: preserveLocalAssistantErrors(
            preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(messages, state.messages), state.messages),
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

/** Best-effort post-turn fallback when the live stream did not carry an answer. */
export async function hydrateStoredSessionTranscript({
  attempts,
  storedSessionId,
  runtimeSessionId,
  storedProfile,
  updateSessionState
}: {
  attempts: number
  storedSessionId: string
  runtimeSessionId: string
  storedProfile: ProfileScope
  updateSessionState: ActiveTranscriptRefreshDeps['updateSessionState']
}): Promise<void> {
  const replay = pendingSessionReplay(runtimeSessionId)

  if (replay && !(await replay)) {
    return
  }

  const messagesAtRequest = $sessionStates.get()[runtimeSessionId]?.messages

  const superseded = () =>
    tileRuntimeOwnsLiveState(runtimeSessionId) ||
    transcriptChangedDuringRead(messagesAtRequest, $sessionStates.get()[runtimeSessionId]?.messages)

  for (let index = 0; index < Math.max(1, attempts); index += 1) {
    if (index > 0) {
      await new Promise(resolve => window.setTimeout(resolve, 250))
    }

    if (superseded()) {
      return
    }

    try {
      const latest = await getLatestSessionMessages(storedSessionId, storedProfile)
      const replayAtReturn = pendingSessionReplay(runtimeSessionId)

      if (replayAtReturn && !(await replayAtReturn)) {
        return
      }

      // This fallback belongs to the completed turn, not any subsequent turn
      // that ran during the read or retry delay. Its todo restore is stale too.
      if (superseded()) {
        return
      }

      // A zero-row page over the populated turn is a transient read, not the
      // answer.
      if (emptyPageOverPopulatedTranscript(latest.messages, $sessionStates.get()[runtimeSessionId], storedSessionId)) {
        continue
      }

      const messages = toChatMessages(latest.messages)
      updateSessionState(
        runtimeSessionId,
        state => ({
          ...state,
          // Keep backfilled pages, un-acked optimistic input and local errors.
          messages: preserveLocalAssistantErrors(
            preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(messages, state.messages), state.messages),
            state.messages
          )
        }),
        storedSessionId
      )
      const restored = todosForHydration(latestSessionTodos(messages))

      if (restored) {
        setSessionTodos(runtimeSessionId, restored)
      } else {
        clearSessionTodos(runtimeSessionId)
      }

      return
    } catch {
      // Best-effort fallback when live stream payloads are empty.
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

  if (!storedSessionId || !runtimeSessionId || busyRef.current || tileRuntimeOwnsLiveState(runtimeSessionId)) {
    return
  }

  const stored = resolveSession(storedSessionId, runtimeSessionId)

  if (!stored) {
    return
  }

  const requestId = requestSequenceRef.current + 1
  requestSequenceRef.current = requestId
  const replay = pendingSessionReplay(runtimeSessionId)

  if (replay && !(await replay)) {
    return
  }

  if (
    requestId !== requestSequenceRef.current ||
    busyRef.current ||
    tileRuntimeOwnsLiveState(runtimeSessionId) ||
    selectedStoredSessionIdRef.current !== storedSessionId ||
    activeSessionIdRef.current !== runtimeSessionId
  ) {
    return
  }

  // Busy at both endpoints can be false even though an entire turn streamed
  // while HTTP was in flight. Never let that older read replace newer text.
  const messagesAtRequest = $sessionStates.get()[runtimeSessionId]?.messages

  try {
    const profileScope: ProfileScope = profileScopeForTranscriptSession(stored)

    const latest = await getLatestSessionMessages(storedSessionId, profileScope)
    const replayAtReturn = pendingSessionReplay(runtimeSessionId)

    if (replayAtReturn && !(await replayAtReturn)) {
      return
    }

    const current = $sessionStates.get()[runtimeSessionId]

    if (
      requestId !== requestSequenceRef.current ||
      busyRef.current ||
      tileRuntimeOwnsLiveState(runtimeSessionId) ||
      transcriptChangedDuringRead(messagesAtRequest, current?.messages) ||
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

    // Same rule as the warm-activation guard (use-session-actions/index.ts):
    // publishing the page would blank the thread and trip the routed loading
    // branch. Bail before the signature write so the next usable page is not
    // deduped away.
    if (emptyPageOverPopulatedTranscript(latest.messages, current, storedSessionId)) {
      return
    }

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
        messages: preserveLocalAssistantErrors(
          preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(messages, state.messages), state.messages),
          state.messages
        )
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
  profileKey = 'default',
  stateAtRequest = $sessionStates.get()
): void {
  const seen = new Set<string>()
  const workingStoredIds = new Set<string>()

  for (const session of response.sessions ?? []) {
    const runtimeSessionId = session.id?.trim()
    const storedSessionId = session.session_key?.trim()
    const needsInput = session.status === 'waiting'
    const working = session.status === 'working' || needsInput

    if (!runtimeSessionId || !storedSessionId) {
      continue
    }

    seen.add(runtimeSessionId)

    if (working) {
      workingStoredIds.add(storedSessionId)
    }

    const existing = $sessionStates.get()[runtimeSessionId]

    // The active-list response is an async snapshot. Stream events can start
    // or finish this turn after the request begins but before its response is
    // applied. In that case the event state is newer: an old idle row must not
    // finish a live turn, and an old working row must not revive a terminal one.
    if (existing !== stateAtRequest[runtimeSessionId]) {
      continue
    }

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

    if (working) {
      // A poll that still lists the turn is an event. Reset the silence clock
      // so a quiet tool call is not settled; a dead backend stops answering
      // this poll and the clock runs out.
      noteSessionEvent(runtimeSessionId)
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

      if (existing !== stateAtRequest[runtimeSessionId]) {
        seen.add(runtimeSessionId)

        continue
      }

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

  // Completions the reconnect reconcile downgraded blind: this snapshot is the
  // terminal fact it lacked. Every parked session not reported working is
  // over, whether or not an earlier poll ever saw its runtime (a turn that
  // started just before the drop was never polled).
  confirmReconnectSettlesExcept(workingStoredIds)
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
  const activeTranscriptRefreshPendingRef = useRef<string | null>(null)
  const activeTranscriptReadRef = useRef<{ sessionKey: string; preservePending: boolean } | null>(null)
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

      if ($busy.get() || tileRuntimeOwnsLiveState(runtimeSessionId)) {
        return
      }

      if (preservePending && activeTranscriptRefreshPendingRef.current === sessionKey) {
        activeTranscriptRefreshPendingRef.current = null
      }

      const read = { sessionKey, preservePending }
      activeTranscriptReadRef.current = read
      void Promise.resolve(refreshActiveTranscript()).finally(() => {
        // A superseded request cannot retire the replacement's observation.
        if (activeTranscriptReadRef.current === read) {
          activeTranscriptReadRef.current = null
        }
      })
    },
    [activeSessionId, activeStoredSessionId, refreshActiveTranscript]
  )

  // eslint-disable-next-line no-restricted-syntax -- request obligations updated by synchronous subscriptions, not mirrored atom values
  useEffect(() => {
    if (gatewayState !== 'open' || !activeSessionId || !activeStoredSessionId) {
      return
    }

    const sessionKey = `${activeStoredSessionId}:${activeSessionId}`
    let disposed = false
    let queued = false

    const isCurrent = () =>
      $activeSessionId.get() === activeSessionId && $selectedStoredSessionId.get() === activeStoredSessionId

    const isLive = () => $busy.get() || tileRuntimeOwnsLiveState(activeSessionId)

    const observe = () => {
      if (!isCurrent()) {
        return
      }

      if (isLive()) {
        const read = activeTranscriptReadRef.current

        if (read?.sessionKey === sessionKey) {
          // Queue once, before idle, even when a whole turn fits between RAFs
          // and the legacy $busy mirror never becomes true.
          activeTranscriptReadRef.current = null

          if (read.preservePending) {
            activeTranscriptRefreshPendingRef.current = sessionKey
          }
        }

        return
      }

      if (queued || activeTranscriptRefreshPendingRef.current !== sessionKey) {
        return
      }

      queued = true
      queueMicrotask(() => {
        queued = false

        // Let the canonical publication finish syncing the view before the
        // retry, and never carry it across navigation/profile/connection scope.
        if (!disposed && isCurrent() && !isLive() && activeTranscriptRefreshPendingRef.current === sessionKey) {
          requestActiveTranscriptRefresh(true)
        }
      })
    }

    const unsubscribeState = $sessionStates.listen(observe)
    const unsubscribeBusy = $busy.listen(observe)

    return () => {
      disposed = true
      unsubscribeState()
      unsubscribeBusy()
      activeTranscriptReadRef.current = null
      activeTranscriptRefreshPendingRef.current = null
    }
  }, [
    activeConnectionId,
    activeGatewayProfile,
    activeSessionId,
    activeStoredSessionId,
    gatewayState,
    requestActiveTranscriptRefresh
  ])

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

  // Reconnect backstop (#94779): turns that finished while the socket was
  // down never replay their sessions.changed tick, so the open transcript
  // stayed stale until the user reopened it. Pull one signature-gated tail on
  // every (re)connect — a no-change read costs nothing. Keyed on the
  // connection, not the session, so a plain session switch adds no read;
  // messaging transcripts already refresh on open in their own effect below.
  useEffect(() => {
    if (gatewayState === 'open' && !activeIsMessaging && activeSessionId && activeStoredSessionId) {
      requestActiveTranscriptRefresh(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- connect-scoped: session deps would fire on every switch
  }, [activeConnectionId, activeGatewayProfile, gatewayState])

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
    let refreshPending = false

    const refreshLiveStatuses = async () => {
      if (cancelled) {
        return
      }

      if (inFlight) {
        refreshPending = true

        return
      }

      inFlight = true
      const stateAtRequest = $sessionStates.get()

      try {
        const response = await requestGateway<LiveSessionStatusResponse>('session.active_list', {})

        if (!cancelled) {
          rehydrateLiveSessionStatuses(response, Date.now(), activeGatewayProfile, stateAtRequest)
        }
      } catch {
        // Older gateways may not expose session.active_list. Live stream events
        // still work as before; leave the current sidebar state untouched —
        // except completions the reconnect reconcile parked awaiting this
        // snapshot: with no snapshot to confirm them they light now (the
        // pre-#113029 behaviour) rather than never.
        if (!cancelled) {
          confirmReconnectSettlesExcept(new Set())
        }
      } finally {
        inFlight = false

        if (refreshPending && !cancelled) {
          refreshPending = false
          void refreshLiveStatuses()
        }
      }
    }

    const unsubscribe = $sessionsChangeTick.listen(() => void refreshLiveStatuses())

    const dispose = visiblePoll(
      changeEventsAvailable ? LIVE_SESSION_STATUS_BACKSTOP_INTERVAL_MS : LIVE_SESSION_STATUS_POLL_INTERVAL_MS,
      () => void refreshLiveStatuses()
    )

    void refreshLiveStatuses()

    return () => {
      cancelled = true
      unsubscribe()
      dispose()
    }
    // Keep the in-flight guard alive across change ticks; a slow response must
    // not create a new request (and invalidate the old result) on every tick.
  }, [activeConnectionId, activeGatewayProfile, changeEventsAvailable, gatewayState, requestGateway])

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

      // Archived sessions use their own capped query. Reload it only while the
      // sidebar is showing that view, matching the on-demand fetch contract.
      if ($sidebarShowArchived.get()) {
        void loadArchivedSessions()
      }

      // The project tree is a grouping of the same stored rows, so a session
      // created/deleted/renamed/re-homed outside this window goes stale in the
      // Projects sidebar without this (#100354). refreshProjectTree() keeps the
      // cached tree on failure, so a not-yet-ready backend costs nothing.
      void refreshProjectTree()

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
