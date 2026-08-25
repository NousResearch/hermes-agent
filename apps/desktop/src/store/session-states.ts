/**
 * MULTI-SESSION VIEW STATE — the reactive face of the per-runtime session
 * cache (`sessionStateByRuntimeIdRef` in use-session-state-cache).
 *
 * The cache already ingests EVERY session's gateway events; only the view
 * was single-session ($messages + the active-id gate). This store mirrors
 * the cache per runtime id so any number of surfaces (session tiles, future
 * pane windows) can each subscribe to one session's state without touching
 * the main chat's `$messages` pipeline — same pattern as `useSessionSlice`
 * over `$todosBySession`, applied to whole `ClientSessionState`s.
 *
 * TILES are the first consumer: sessions opened side-by-side with the main
 * thread, each in its own layout-tree pane. `$sessionTiles` holds the
 * stored-session ids (persisted — tiles survive restarts); the wiring layer
 * owns resume/submit (it has the gateway + cache internals) and registers
 * itself here as the delegate so tile UI stays dependency-light.
 */

import { LOCAL_CONNECTION_ID, registryBackendScopeKey } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import type { ClientSessionState } from '@/app/types'
import { findGroup, findGroupOfPane, type LayoutNode } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $layoutTree,
  focusedSessionTabAnchor,
  moveTreePane,
  noteActiveTreeGroup,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import { $workspaceMode, resolveRememberedActivePane, workspaceScopeKey } from '@/components/pane-shell/workspace-scope'
import type { WorkspaceMode } from '@/contrib/types'
import { stableArray } from '@/lib/stable-array'
import { readJson, writeJson } from '@/lib/storage'
import type { SessionInfo } from '@/types/hermes'

import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import { clearAllProviderWaits, clearSessionProviderWait } from './provider-wait'
import {
  $activeSessionId,
  $connection,
  $lastReadAtBySessionId,
  $selectedStoredSessionId,
  $sessions,
  clearReadBaseline,
  getSessionOwnerHint,
  knownSessionOwner,
  lineageAliases,
  markSessionRead,
  ownerLookupSessionRows,
  sessionMatchesStoredId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setSessions
} from './session'
import { assertSessionOwnerResolved } from './session-owner-resolution'
import { requestForSessionProfile, type SessionOwnerRoute, type SessionOwnerScope } from './session-request-router'
import { ackStoredSessionId, markSessionUnreadFinished } from './session-unread'
import { isBrowserWindow, isSecondaryWindow } from './windows'

// ---------------------------------------------------------------------------
// Reactive per-runtime session state (view mirror of the wiring cache).
// ---------------------------------------------------------------------------

export const $sessionStates = atom<Record<string, ClientSessionState>>({})

// ---------------------------------------------------------------------------
// Event-source scopes: which registry connection's socket delivered a runtime
// session's events. Working/attention membership alone is profile-blind — two
// connected gateways can both expose a 'default' profile, so the gateway
// keep-set (pruneSecondaryGateways) must key live work by the composite
// (connectionId, profile) scope, not the bare profile name. Recorded at
// event fan-in (use-gateway-boot); local/primary events carry no connectionId
// and record nothing, so single-source behavior is untouched.
// ---------------------------------------------------------------------------

const sessionScopeByRuntimeId = new Map<string, string>()

export function recordSessionEventScope(event: { connectionId?: string; profile?: string; session_id?: string }): void {
  if (event.session_id && event.connectionId) {
    sessionScopeByRuntimeId.set(event.session_id, registryBackendScopeKey(event.connectionId, event.profile))
  }
}

/** Composite scopes of registry-sourced sessions that are live (busy or
 * waiting on input) — the (connectionId, profile) half of the gateway
 * keep-set. Local-source live work keeps flowing through profile names. */
export function liveSessionScopes(): Set<string> {
  const scopes = new Set<string>()

  for (const [runtimeId, state] of Object.entries($sessionStates.get())) {
    if (!state || (!state.busy && !state.needsInput)) {
      continue
    }

    const scope = sessionScopeByRuntimeId.get(runtimeId)

    if (scope) {
      scopes.add(scope)
    }
  }

  return scopes
}

// ── Owner hold across the create → foreground gap ───────────────────────────
// A routed session.create returns a stored id on the owner's socket, but the
// surface that will PIN that socket (the selected primary thread, or a tile)
// is published later and asynchronously: navigate → route effect →
// $selectedStoredSessionId, or openSessionTile → $sessionTiles. In that gap
// the entry has no active request, is not yet foreground-bound and, if the
// user switched source meanwhile, is not the active key either — so the
// live-work pruner or a refcount-0 lease release could close the socket that
// holds the just-minted runtime before the first prompt.submit. The hold
// names the owner in foregroundSessionScopes from the moment the create
// returns until the foreground publication takes over (the stored id becomes
// selected or tiled), the caller releases it (failed create / drift close),
// or a bounded TTL expires — nothing latches.
const SESSION_OWNER_HOLD_TTL_MS = 60_000

const sessionOwnerHolds = new Map<
  string,
  { owner: SessionOwnerScope; timer: ReturnType<typeof setTimeout>; until: number }
>()

export const $sessionOwnerHoldRevision = atom(0)

function bumpSessionOwnerHoldRevision(): void {
  $sessionOwnerHoldRevision.set($sessionOwnerHoldRevision.get() + 1)
}

function forgetSessionOwnerHold(storedSessionId: string, publish: boolean): boolean {
  const hold = sessionOwnerHolds.get(storedSessionId)

  if (!hold) {
    return false
  }

  clearTimeout(hold.timer)
  sessionOwnerHolds.delete(storedSessionId)

  if (publish) {
    bumpSessionOwnerHoldRevision()
  }

  return true
}

export function holdSessionOwnerUntilForeground(storedSessionId: string, owner: SessionOwnerScope): () => void {
  const id = storedSessionId.trim()

  if (!id || !owner) {
    return () => undefined
  }

  forgetSessionOwnerHold(id, false)
  const until = Date.now() + SESSION_OWNER_HOLD_TTL_MS
  const timer = setTimeout(() => releaseSessionOwnerHold(id), SESSION_OWNER_HOLD_TTL_MS)

  sessionOwnerHolds.set(id, { owner, timer, until })
  bumpSessionOwnerHoldRevision()

  return () => releaseSessionOwnerHold(id)
}

export function releaseSessionOwnerHold(storedSessionId: string): void {
  forgetSessionOwnerHold(storedSessionId.trim(), true)
}

/** @internal Tests. */
export function _resetSessionOwnerHoldsForTests(): void {
  const hadHolds = sessionOwnerHolds.size > 0

  for (const hold of sessionOwnerHolds.values()) {
    clearTimeout(hold.timer)
  }

  sessionOwnerHolds.clear()

  if (hadHolds) {
    bumpSessionOwnerHoldRevision()
  }
}

/**
 * Registry scopes owned by an open foreground surface, when known.
 *
 * The secondary-gateway pruner normally keeps only busy/needs-input work. A
 * source switch briefly changes the active gateway before an idle conversation
 * is cleared, so the primary runtime must survive that handoff. Open panes have
 * the same ownership contract: a non-focused idle tile is still user-visible
 * state and must not be evicted just because another pane has focus. Prefer the
 * live event scope, with the tile's persisted route as the pre-bind fallback.
 *
 * A just-created session's owner is named by its create → foreground hold
 * (holdSessionOwnerUntilForeground) until the selected/tiled publication or
 * a bounded TTL retires it, so nothing can close the socket that minted the
 * runtime before the first prompt lands.
 */
export function foregroundSessionScopes(): Set<string> {
  const scopes = new Set<string>()

  const addRuntimeScope = (runtimeId: string | undefined) => {
    const scope = runtimeId ? sessionScopeByRuntimeId.get(runtimeId) : undefined

    if (scope) {
      scopes.add(scope)
    }
  }

  const addRouteScope = (route: SessionOwnerRoute | undefined) => {
    const connectionId = route?.connectionId?.trim()
    const profile = route?.profile?.trim()

    if (connectionId && profile) {
      scopes.add(registryBackendScopeKey(connectionId, profile))
    }
  }

  addRuntimeScope($activeSessionId.get() ?? undefined)

  for (const tile of $sessionTiles.get()) {
    addRuntimeScope(tile.runtimeId)
    addRouteScope(tile.ownerRoute)
  }

  // Create → foreground holds. A hold whose scope the rungs above already
  // name (the runtime's event scope once selected, a mounted tile's route) is
  // covered and retires; an expired one retires too.
  const now = Date.now()

  for (const [storedSessionId, hold] of [...sessionOwnerHolds]) {
    const scope =
      typeof hold.owner === 'string'
        ? normalizeProfileKey(hold.owner)
        : hold.owner?.connectionId?.trim()
          ? registryBackendScopeKey(hold.owner.connectionId.trim(), normalizeProfileKey(hold.owner.profile))
          : null

    if (!scope || hold.until <= now || scopes.has(scope)) {
      // This recompute was already triggered by the covering publication (or
      // is itself observing expiry), so avoid recursively publishing.
      forgetSessionOwnerHold(storedSessionId, false)

      continue
    }

    scopes.add(scope)
  }

  return scopes
}

// Stored session ids whose authoritative state is still busy, but whose
// runtime has produced no state publish for the watchdog window. Silence is
// not completion: long tool calls can legitimately stay quiet, so this is a
// presentation hint and never mutates the backend-derived busy state.
export const $stalledSessionIds = atom<string[]>([])

export function setSessionStalled(storedSessionId: string | null | undefined, stalled: boolean) {
  if (!storedSessionId) {
    return
  }

  const current = $stalledSessionIds.get()
  const present = current.includes(storedSessionId)

  if (stalled && !present) {
    $stalledSessionIds.set([...current, storedSessionId])
  } else if (!stalled && present) {
    $stalledSessionIds.set(current.filter(id => id !== storedSessionId))
  }
}

// --- Watchdog: marks busy sessions quiet after a long stream silence -------
// Tuned against what this app actually does rather than a round number: a
// typecheck or a full test run here goes quiet for minutes at a stretch and is
// perfectly healthy, so anything under ~4 min would paint normal work as
// suspect. Eight minutes was the other failure — longer than a user is willing
// to sit and wonder, so the hint arrived after they had already given up on it.
export const SESSION_WATCHDOG_TIMEOUT_MS = 5 * 60 * 1000
const sessionWatchdogTimers = new Map<string, ReturnType<typeof setTimeout>>()

function armWatchdog(runtimeId: string) {
  const existing = sessionWatchdogTimers.get(runtimeId)

  if (existing) {
    clearTimeout(existing)
  }

  sessionWatchdogTimers.set(
    runtimeId,
    setTimeout(() => {
      sessionWatchdogTimers.delete(runtimeId)
      const current = $sessionStates.get()[runtimeId]

      if (current?.busy) {
        setSessionStalled(current.storedSessionId, true)
      }
    }, SESSION_WATCHDOG_TIMEOUT_MS)
  )
}

function clearWatchdog(runtimeId: string) {
  const t = sessionWatchdogTimers.get(runtimeId)

  if (t) {
    clearTimeout(t)
    sessionWatchdogTimers.delete(runtimeId)
  }
}

// --- Settle grace: keeps a just-finished session in the sidebar merge set ---
const SESSION_SETTLE_GRACE_MS = 30 * 1000
const settledExpiry = new Map<string, number>()

function markSettled(storedId: string) {
  settledExpiry.set(storedId, Date.now() + SESSION_SETTLE_GRACE_MS)
}

function clearSettled(storedId: string) {
  settledExpiry.delete(storedId)
}

/** Stored ids whose turn ended within the grace window. Prunes expired. */
export function getRecentlySettledSessionIds(now: number = Date.now()): string[] {
  const live: string[] = []

  for (const [id, expiry] of settledExpiry) {
    if (expiry > now) {
      live.push(id)
    } else {
      settledExpiry.delete(id)
    }
  }

  return live
}

// --- Transition detection (called automatically from publishSessionState) ---
function handleTransition(previous: ClientSessionState | null, next: ClientSessionState, runtimeId: string) {
  // Compression id rotation: signal the route-follow effect with enough
  // provenance (previous id + runtime) that the consumer can reject the event
  // if the user navigated elsewhere before React handled it. A bare next id
  // could let a background session's delayed rotation steal the foreground
  // route.
  if (previous?.storedSessionId && next.storedSessionId && previous.storedSessionId !== next.storedSessionId) {
    if (runtimeId === $activeSessionId.get()) {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: next.storedSessionId,
        previousStoredSessionId: previous.storedSessionId,
        runtimeSessionId: runtimeId
      })
    }

    clearSettled(previous.storedSessionId)
    setSessionStalled(previous.storedSessionId, false)
  }

  // Every busy publish is stream activity: clear the quiet hint and restart
  // the silence window. A real terminal transition clears both the timer and
  // any hint, but only that authoritative transition clears working/busy.
  if (next.busy) {
    setSessionStalled(next.storedSessionId, false)
    armWatchdog(runtimeId)
  } else {
    clearWatchdog(runtimeId)
    setSessionStalled(next.storedSessionId, false)
    setSessionStalled(previous?.storedSessionId, false)
  }

  const storedId = next.storedSessionId

  if (!storedId) {
    return
  }

  const wasWorking = previous?.busy ?? false

  if (next.busy && !wasWorking) {
    clearSettled(storedId)
    // A NEW turn is starting: the read baseline guarded the PREVIOUS
    // completion's re-asserts. Dropping it here means this turn's finish
    // re-lights even if it lands within the same millisecond as the last
    // read (same-tick submit → finish in tests and fast local models).
    clearReadBaseline(storedId)
  } else if (!next.busy && wasWorking) {
    markSettled(storedId)

    // FOCUSED, not selected: a session finishing in the tile the user is
    // watching is already seen, and a tile is never the primary selection.
    if (storedId !== $focusedStoredSessionId.get()) {
      // Re-light only genuinely new completions: if the user already viewed
      // this session (or its family) at or after this settle moment, a
      // re-assert of the same completion must not re-arm the dot. `-1` for
      // "never read" (not `0`) so fake-timer tests pinned to t=0 still light.
      const lastReadAt = $lastReadAtBySessionId.get()[storedId] ?? -1

      if (Date.now() > lastReadAt) {
        // Flags the transient atom AND persists a marker, so the green dot
        // survives an app restart (see session-unread.ts).
        markSessionUnreadFinished(storedId)
      }
    }
  }
}

/** Is any surface on THIS window still holding the runtime — the primary view
 *  or an open tile? (A tile mid-resume references by stored id only; its
 *  runtime binding is patched in after `resumeTile` returns.) */
function runtimeReferenced(runtimeId: string, storedSessionId: null | string): boolean {
  if (runtimeId === $activeSessionId.get()) {
    return true
  }

  return $sessionTiles
    .get()
    .some(t => t.runtimeId === runtimeId || (storedSessionId !== null && t.storedSessionId === storedSessionId))
}

/** A state no surface needs anymore: its turn is over (not busy, not waiting
 *  on the user) and neither the primary view nor any tile holds the runtime.
 *  `needsInput` states stay — the sidebar's attention dot reads them. */
function evictable(runtimeId: string, state: ClientSessionState): boolean {
  return (
    !state.busy && !state.needsInput && !state.awaitingResponse && !runtimeReferenced(runtimeId, state.storedSessionId)
  )
}

/** Publish one session's state. Automatically fires transition side-effects
 *  (watchdog arm/disarm, settle grace, unread marker, compression id rotation)
 *  by diffing previous vs next — callers never need to manually call a
 *  transition handler.
 *
 *  Skips the publish when the new state is identical to the existing one
 *  (same reference) to avoid churning `$sessionStates` on periodic
 *  `session.info` heartbeats that carry no change — otherwise every ~1/s
 *  heartbeat creates a new Record spread, triggering computed atoms
 *  ($workingSessionIds, $attentionSessionIds) and their subscribers
 *  unnecessarily. The runtime-id→state cache (sessionStateByRuntimeIdRef)
 *  is updated independently by the caller, so the visual path stays live
 *  without the store churn.
 *
 *  A settled state nothing references releases its transcript instead of
 *  republishing it. Gateway events keep flowing for sessions whose tile was
 *  closed mid-turn, and parking each one's full transcript here forever is the
 *  leak that made the app crawl after a day of tile use. Transition side
 *  effects still fire, so lightweight status and the unread dot survive. A
 *  FIRST publish always lands in full because a resume can publish its idle
 *  state a beat before `$activeSessionId` / the tile binding points at it. */
export function publishSessionState(runtimeId: string, state: ClientSessionState) {
  const current = $sessionStates.get()
  const prev = current[runtimeId] ?? null

  if (prev === state) {
    return
  }

  if (prev && evictable(runtimeId, state)) {
    handleTransition(prev, state, runtimeId)
    releaseSessionTranscript(runtimeId, state)

    return
  }

  $sessionStates.set({ ...current, [runtimeId]: state })
  handleTransition(prev, state, runtimeId)
}

/** Keep the cheap status projection for a cold session while releasing its
 * transcript. Unread completion is stored separately, so it survives too. */
export function releaseSessionTranscript(runtimeId: string, state?: ClientSessionState) {
  const current = $sessionStates.get()

  if (!(runtimeId in current)) {
    return
  }

  const retained = state ?? current[runtimeId]

  // Older persisted snapshots can contain an undefined state or omit the
  // messages field. Treat either shape as already cold instead of throwing
  // while memory pressure is being relieved.
  if (!retained) {
    return
  }

  const lightweight =
    Array.isArray(retained.messages) && retained.messages.length === 0 ? retained : { ...retained, messages: [] }

  $sessionStates.set({ ...current, [runtimeId]: lightweight })
}

export function dropSessionState(runtimeId: string) {
  // Disarm the watchdog — a dropped runtime must not fire a stale clear later.
  // Settle-grace entries are keyed by stored id and self-expire; leave them so
  // a just-finished session's row survives merge eviction even if its tile or
  // cached runtime is dropped in the meantime.
  clearWatchdog(runtimeId)
  clearSessionProviderWait(runtimeId)
  sessionScopeByRuntimeId.delete(runtimeId)

  const current = $sessionStates.get()
  setSessionStalled(current[runtimeId]?.storedSessionId, false)

  if (!(runtimeId in current)) {
    return
  }

  const { [runtimeId]: _dropped, ...rest } = current
  $sessionStates.set(rest)
}

/** Drop every cached session state — used on soft gateway-mode apply so the
 *  computed working / attention sets drain to empty alongside the session list.
 *  Also disarms every watchdog timer and drops all settle-grace entries: a
 *  wiped gateway's sessions must not fire stale clears or linger in the
 *  sidebar merge keep-set after the switch. */
export function clearAllSessionStates() {
  for (const timer of sessionWatchdogTimers.values()) {
    clearTimeout(timer)
  }

  sessionWatchdogTimers.clear()
  settledExpiry.clear()
  clearAllProviderWaits()
  sessionScopeByRuntimeId.clear()
  $stalledSessionIds.set([])
  $sessionStates.set({})
}

/** Downgrade cached busy/awaiting states after a gateway reconnect.
 *
 *  A respawned backend re-mints runtime ids (the same fact that drives
 *  resetTileRuntimeBindings), so a pre-reconnect `busy` can never receive its
 *  terminal `busy: false` publish — the runtime id it would arrive under is
 *  dead. Left alone, that state keeps its session in $workingSessionIds
 *  forever: the sidebar running arc and agents-panel "running" chrome lie for
 *  hours after the turn actually ended (#53902, #73082 — stale-flag half).
 *
 *  `scope` picks which socket's sessions to reconcile, keyed by the event-
 *  source scope recorded at fan-in: a SECONDARY (registry) reconnect passes
 *  its composite scope and touches only runtimes that arrived on that socket;
 *  the PRIMARY reconnect passes undefined and touches only scope-less
 *  runtimes (primary/local events record no scope). Neither can clear live
 *  work riding a different, still-healthy connection.
 *
 *  Direction of failure is deliberate: a turn that IS still live (transient
 *  socket blip, same backend) re-asserts busy on its next event or inflight
 *  snapshot within a beat, so at worst its arc blinks once. A dead turn's
 *  state, by contrast, would never clear on its own. `needsInput` is left
 *  untouched — a blocking prompt is the one claim the user must explicitly
 *  answer, and post-reconnect refresh re-asserts or retires it via its own
 *  path. Transition side-effects run through publishSessionState, so
 *  watchdogs disarm, stall hints drop, and settle/unread bookkeeping stays
 *  consistent.
 *
 *  The downgrade goes through the delegate's `retireBusyClaim` (the wiring
 *  cache's updateSessionState), not straight into this mirror: the claim has
 *  four holders — wiring cache, mirror, the focused view's draft latches,
 *  busyRef — and retiring only the mirror left Send silently no-oping behind
 *  a stale busy until restart (#93059). The mirror publish stays as the
 *  fallback for runtimes the cache never held (background-sync rows, no
 *  wiring mounted). A PRIMARY reconcile also clears the focused draft
 *  latches, which outlive the state they mirrored; a scoped one leaves them
 *  alone — a background socket says nothing about the primary composer. */
export function reconcileBusyStatesOnReconnect(scope?: string) {
  const states = $sessionStates.get()

  for (const [runtimeId, state] of Object.entries(states)) {
    if (!state || (!state.busy && !state.awaitingResponse)) {
      continue
    }

    const recorded = sessionScopeByRuntimeId.get(runtimeId)

    if (scope === undefined ? recorded !== undefined : recorded !== scope) {
      continue
    }

    sessionTileDelegate()?.retireBusyClaim?.(runtimeId)

    // Re-read — the write path may have republished (and released) this entry.
    const published = $sessionStates.get()[runtimeId]

    if (published?.busy || published?.awaitingResponse) {
      publishSessionState(runtimeId, { ...published, awaitingResponse: false, busy: false })
    }
  }

  if (scope === undefined) {
    setBusy(false)
    setAwaitingResponse(false)
  }
}

// Derived per-session status sets — pure projections of `$sessionStates` (which
// holds `busy`/`needsInput` per runtime), keeping the data flow one-directional:
// gateway event → cache → $sessionStates → computed views.
//
// Perf: `$sessionStates` is republished on EVERY message delta (tens/sec during
// a turn), but these sets only change on busy/needsInput edges. `stableArray`
// keeps the prior reference when membership is unchanged so `computed` skips the
// emit — otherwise the whole sidebar + every row re-renders per token.
// Published under every id the conversation answers to, not just its current
// tip: consumers hold whichever id they were created with, and compression
// rotates the tip out from under them (see lineageAliases).
//
// A conversation that has not been persisted yet has no stored id at all, and
// dropping it here is what left the FIRST turn of a new chat with no running
// indicator anywhere — no dot, no row arc — for as long as it took the backend
// to hand one back. Its runtime id is the right fallback because until a stored
// id exists the two are the same value (submit.ts: "an unpersisted
// conversation's queue key IS its runtime id"), so the row matches; once a
// session is persisted its runtime id is nobody's key and the fallback is inert.
const storedIds = (
  states: Record<string, ClientSessionState>,
  sessions: readonly SessionInfo[],
  pred: (s: ClientSessionState) => boolean
) => {
  const ids = new Set<string>()

  for (const [runtimeId, state] of Object.entries(states)) {
    if (!pred(state)) {
      continue
    }

    for (const alias of lineageAliases(state.storedSessionId ?? runtimeId, sessions)) {
      ids.add(alias)
    }
  }

  return [...ids]
}

let workingIds: readonly string[] = []
export const $workingSessionIds = computed(
  [$sessionStates, $sessions],
  (states, sessions) =>
    (workingIds = stableArray(
      workingIds,
      storedIds(states, sessions, s => s.busy)
    ))
)

let attentionIds: readonly string[] = []
export const $attentionSessionIds = computed(
  [$sessionStates, $sessions],
  (states, sessions) =>
    (attentionIds = stableArray(
      attentionIds,
      storedIds(states, sessions, s => s.needsInput)
    ))
)

// An open session nothing has ever been sent to — the ⌘T tab whose backend
// session exists but is unlisted, or a tile still waiting on its first send.
// `blankDraftTile`'s predicate, read as a status rather than as a slot to spend.
//
// The row's own `message_count` is the tiebreaker, and it is load-bearing: a
// session RESUMING also holds an empty message list for the moment between
// binding its runtime and loading its transcript, and calling that a draft
// would flash the wrong mark on a conversation with years of history in it.
let draftIds: readonly string[] = []
export const $draftSessionIds = computed([$sessionStates, $sessions], (states, sessions) => {
  const unsent = (state: ClientSessionState) => {
    if (state.busy || state.messages.length > 0) {
      return false
    }

    const storedId = state.storedSessionId

    // No stored id is the ⌘T tab that hasn't reached the backend yet: a draft
    // by definition, and no row to consult. Asking anyway would match a row on
    // an empty lineage root.
    if (!storedId) {
      return true
    }

    const row = sessions.find(session => sessionMatchesStoredId(session, storedId))

    return !row || row.message_count === 0
  }

  return (draftIds = stableArray(draftIds, storedIds(states, sessions, unsent)))
})

// ---------------------------------------------------------------------------
// Session tiles.
// ---------------------------------------------------------------------------

/** Edge a tile docks against main when it first joins the tree. Shared by
 *  session tiles and route (page) tiles. */
export type SplitDir = 'bottom' | 'left' | 'right' | 'top'

/** Where a tile lands on adoption: an edge split, or `center` = stack into
 *  the anchor's zone as a tab (a drop on the zone's tab strip). */
export type TileDock = 'center' | SplitDir

export interface SessionTile {
  /** Stored session id — the durable identity (runtime ids are ephemeral). */
  storedSessionId: string
  /** Dock against `anchor` on adoption (default right; center = stack). */
  dir?: TileDock
  /** Pane to dock against (a drop's target zone) — default the workspace.
   *  Persisted so a restart re-docks in place; a stale id falls back to the
   *  workspace (findGroupOfPane misses → the move is skipped). */
  anchor?: string
  /** Center docks: stack BEFORE this pane id (`null`/omitted = append) — the
   *  strip divider's slot. Persisted, like `anchor`; a stale id appends. */
  before?: null | string
  /** Live runtime id once the tile's resume has bound one. */
  runtimeId?: string
  /** Resume failed terminally (shown in the tile; retryable). */
  error?: string
  /** Presentation workspace this tab belongs to. Missing legacy values are Sessions. */
  workspaceMode?: WorkspaceMode
  /** Exact opaque owner key for Bot Mode tabs. */
  workspaceOwnerKey?: string
  /** Credential-free exact route used to resume this tab after relaunch. */
  ownerRoute?: SessionOwnerRoute
  /** Stable title for hidden relationship chats absent from the Sessions list. */
  workspaceTabTitle?: string
}

export interface SessionTileWorkspaceScope {
  ownerRoute?: SessionOwnerRoute
  workspaceMode: WorkspaceMode
  workspaceOwnerKey?: string
  workspaceTabTitle?: string
}

// Tiles are persisted PER PROFILE: a session belongs to one profile, and the
// single live gateway is scoped to one profile at a time, so a tile only makes
// sense while its profile is active. Switching profiles swaps the visible set
// (and drops runtime bindings so each tile re-resumes against the now-current
// gateway — which also settles the "tile resumes against the wrong backend" and
// "stale runtime after respawn" bugs by construction).
const TILES_KEY = 'hermes.desktop.sessionTiles.v2'
const LEGACY_TILES_KEY = 'hermes.desktop.sessionTiles.v1'
const TILE_PANE_PREFIX = 'session-tile:'
const BOTS_TILE_BUCKET = '__bots_workspace__'

/** Persisted placement — `dir` + strip slot (`before`) + dock `anchor` so a
 *  restart / profile swap re-adopts tiles in the same order, not all stacked
 *  right of workspace. */
type StoredTile = Pick<
  SessionTile,
  | 'anchor'
  | 'before'
  | 'dir'
  | 'ownerRoute'
  | 'storedSessionId'
  | 'workspaceMode'
  | 'workspaceOwnerKey'
  | 'workspaceTabTitle'
>

const toStored = (t: SessionTile): StoredTile => ({
  anchor: t.anchor,
  before: t.before,
  dir: t.dir,
  ...(t.ownerRoute ? { ownerRoute: t.ownerRoute } : {}),
  storedSessionId: t.storedSessionId,
  ...(t.workspaceMode ? { workspaceMode: t.workspaceMode } : {}),
  ...(t.workspaceOwnerKey ? { workspaceOwnerKey: t.workspaceOwnerKey } : {}),
  ...(t.workspaceTabTitle ? { workspaceTabTitle: t.workspaceTabTitle } : {})
})

function parseTileList(value: unknown): StoredTile[] {
  return Array.isArray(value)
    ? value
        .filter((t): t is SessionTile => Boolean(t && typeof (t as SessionTile).storedSessionId === 'string'))
        .map(t => {
          const raw = t as SessionTile

          return {
            anchor: typeof raw.anchor === 'string' ? raw.anchor : undefined,
            before: typeof raw.before === 'string' || raw.before === null ? raw.before : undefined,
            dir: raw.dir,
            ownerRoute:
              raw.ownerRoute &&
              typeof raw.ownerRoute.connectionId === 'string' &&
              typeof raw.ownerRoute.profile === 'string'
                ? {
                    connectionId: raw.ownerRoute.connectionId,
                    mode: raw.ownerRoute.mode,
                    profile: raw.ownerRoute.profile,
                    ...(typeof raw.ownerRoute.targetProfile === 'string'
                      ? { targetProfile: raw.ownerRoute.targetProfile }
                      : {})
                  }
                : undefined,
            storedSessionId: raw.storedSessionId,
            workspaceMode: raw.workspaceMode === 'bots' ? 'bots' : 'sessions',
            workspaceOwnerKey:
              raw.workspaceMode === 'bots' && typeof raw.workspaceOwnerKey === 'string'
                ? raw.workspaceOwnerKey
                : undefined,
            workspaceTabTitle: typeof raw.workspaceTabTitle === 'string' ? raw.workspaceTabTitle : undefined
          }
        })
    : []
}

function loadTilesByProfile(): Record<string, StoredTile[]> {
  const byProfile: Record<string, StoredTile[]> = {}
  const parsed = readJson<unknown>(TILES_KEY)

  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    for (const [profile, list] of Object.entries(parsed as Record<string, unknown>)) {
      const tiles = parseTileList(list)
      const key = profile === BOTS_TILE_BUCKET ? BOTS_TILE_BUCKET : normalizeProfileKey(profile)

      if (tiles.length > 0) {
        const sessionTiles = tiles.filter(tile => tile.workspaceMode !== 'bots')
        const botTiles = tiles.filter(tile => tile.workspaceMode === 'bots')

        if (sessionTiles.length > 0) {
          byProfile[key] = [...(byProfile[key] ?? []), ...sessionTiles]
        }

        if (botTiles.length > 0) {
          byProfile[BOTS_TILE_BUCKET] = [...(byProfile[BOTS_TILE_BUCKET] ?? []), ...botTiles]
        }
      }
    }
  }

  // Migrate a v1 flat list into the default profile, then retire the key.
  const legacy = parseTileList(readJson<unknown>(LEGACY_TILES_KEY))

  if (legacy.length > 0) {
    const key = normalizeProfileKey('default')
    const sessionTiles = legacy.filter(tile => tile.workspaceMode !== 'bots')
    const botTiles = legacy.filter(tile => tile.workspaceMode === 'bots')

    byProfile[key] = [...(byProfile[key] ?? []), ...sessionTiles]
    byProfile[BOTS_TILE_BUCKET] = [...(byProfile[BOTS_TILE_BUCKET] ?? []), ...botTiles]
  }

  if (byProfile[BOTS_TILE_BUCKET]?.length) {
    byProfile[BOTS_TILE_BUCKET] = [
      ...new Map(byProfile[BOTS_TILE_BUCKET].map(tile => [tile.storedSessionId, tile])).values()
    ]
  }

  writeJson(LEGACY_TILES_KEY, null)

  return byProfile
}

const tilesByProfile = loadTilesByProfile()
// Keyed by the GATEWAY profile: the rail's profile switch is a soft swap
// ($activeGatewayProfile moves, no reload) — $activeProfile mirrors the
// window's primary backend and never changes on a rail switch, so keying on
// it left the previous profile's tiles registered (phantom "Session" tabs).
const profileKey = () => normalizeProfileKey($activeGatewayProfile.get())

// Runtime ids are process-scoped — never trust a persisted one, so the live
// atom hydrates from the stored (runtime-less) tiles for the active profile.
// A secondary window (single-chat pop-out) shows ONLY its routed session — no
// tiles, and no repopulation on a profile switch.
export const $sessionTiles = atom<SessionTile[]>(
  isSecondaryWindow() || isBrowserWindow()
    ? []
    : [...(tilesByProfile[profileKey()] ?? []), ...(tilesByProfile[BOTS_TILE_BUCKET] ?? [])]
)

function persistTiles() {
  // Shares the origin's storage; a secondary / browser pop-out holds no tiles,
  // so a write back would only wipe the primary's set.
  if (isSecondaryWindow() || isBrowserWindow()) {
    return
  }

  writeJson(TILES_KEY, Object.keys(tilesByProfile).length === 0 ? null : tilesByProfile)
}

function saveTiles(tiles: SessionTile[]) {
  const stored = tiles.map(toStored)
  const sessionTiles = stored.filter(tile => tile.workspaceMode !== 'bots')
  const botTiles = stored.filter(tile => tile.workspaceMode === 'bots')

  if (sessionTiles.length > 0) {
    tilesByProfile[profileKey()] = sessionTiles
  } else {
    delete tilesByProfile[profileKey()]
  }

  if (botTiles.length > 0) {
    tilesByProfile[BOTS_TILE_BUCKET] = botTiles
  } else {
    delete tilesByProfile[BOTS_TILE_BUCKET]
  }

  persistTiles()
  $sessionTiles.set(tiles)
}

// Profile switch: surface the new profile's tiles with runtime ids cleared so
// they re-resume against the now-current gateway. (Fires immediately on
// subscribe; harmless — the init value already matches.) A secondary window
// never carries tiles, so it stays out of this entirely.
if (!isSecondaryWindow() && !isBrowserWindow()) {
  $activeGatewayProfile.subscribe(() => {
    $sessionTiles.set([...(tilesByProfile[profileKey()] ?? []), ...(tilesByProfile[BOTS_TILE_BUCKET] ?? [])])
  })
}

export function patchSessionTile(storedSessionId: string, patch: Partial<SessionTile>) {
  saveTiles($sessionTiles.get().map(t => (t.storedSessionId === storedSessionId ? { ...t, ...patch } : t)))
}

export function sessionTileOwnerRoute(storedSessionId: string): SessionOwnerRoute | undefined {
  return $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)?.ownerRoute
}

/**
 * Gateway keep-set scopes for currently open tiles. Bot chats (and any other
 * owner-routed tile) hold a secondary socket even while chrome stays on the
 * launch profile; without these keys, idle prune closes that socket and the
 * tile's resume/unbind loop spins forever. Local routes contribute both the
 * bare profile (openGatewayForProfile) and the explicit `conn:local::…` key
 * (openGatewayForAgent). Remote routes contribute only the composite key so
 * a homelab tile cannot pin another source's same-named profile.
 */
export function openTileGatewayScopes(): Set<string> {
  const scopes = new Set<string>()

  for (const tile of $sessionTiles.get()) {
    const route = tile.ownerRoute

    if (!route) {
      continue
    }

    const profile = normalizeProfileKey(route.profile)
    const connectionId = String(route.connectionId ?? '').trim()
    const localRoute = !connectionId || connectionId === LOCAL_CONNECTION_ID || route.mode === 'local'

    if (localRoute) {
      scopes.add(profile)
    }

    if (connectionId) {
      scopes.add(registryBackendScopeKey(connectionId, profile))
    }
  }

  return scopes
}

/**
 * Sync owner resolution for a session id that may be a RUNTIME or a STORED id.
 * Tile route first (exact connectionId+profile, survives relaunch), then the
 * exact unique owner hint (stamped when a routed create returns / at open
 * time; persisted), then the session row's owner (an exact route when the row
 * is connection-tagged, else its bare profile, else the hint's profile). The
 * row rung searches every source-scoped slice (recents, cron, messaging), not
 * just recents — a cron session's approval.respond used to find no owner here
 * and fail closed on registry-topology installs even though its row (with its
 * `profile` stamp) was already loaded for the sidebar's cron section. The
 * hint outranks the row for the same reason as contrib/wiring's ladder: a
 * row can be stamped from the ambient profile and carries no connection.
 * Returns undefined when no owner is known — the caller fails closed
 * (assertSessionOwnerResolved), never falls to "active".
 */
export function knownOwnerForSession(sessionId: null | string | undefined): SessionOwnerScope {
  if (!sessionId) {
    return undefined
  }

  const storedSessionId = storedSessionIdForRuntimeId(sessionId) ?? sessionId

  return (
    sessionTileOwnerRoute(storedSessionId) ??
    getSessionOwnerHint(storedSessionId) ??
    knownSessionOwner(ownerLookupSessionRows(), storedSessionId)
  )
}

/**
 * Whether the connection that OWNS `sessionId` is remote — never the ambient
 * `$connection`. A session tied to a registered secondary connection (Bot
 * Mode, the unified Sessions list) can differ from whichever connection the
 * window currently shows; its RPCs already route to their own owner via
 * `requestForSessionProfile`, but a caller that instead reads ambient mode to
 * decide image.attach vs image.attach_bytes ships a client-local path to a
 * remote backend that can't resolve it (#94640). A bare profile name (no
 * connectionId) is a pool profile of the ambient connection, so ambient mode
 * still applies there.
 */
export function isSessionRemote(sessionId: null | string | undefined): boolean {
  const owner = knownOwnerForSession(sessionId)

  if (owner && typeof owner === 'object' && owner.mode) {
    return owner.mode === 'remote'
  }

  return $connection.get()?.mode === 'remote'
}

/**
 * Dispatch a session-scoped RPC through the OWNER of `sessionId` (tile route →
 * hint → connection-tagged row / known profile). This is the client half of
 * #91684: approval.respond (and siblings) sent on the ambient socket land on
 * whatever backend is active, which for a cross-profile session is a backend
 * that never held the approval. An UNKNOWN owner fails closed with an
 * explicit SessionOwnerResolutionError unless the ambient gateway is provably
 * the only backend (legacy single-profile, no registry source).
 */
export function requestForOwnedSession<T>(
  sessionId: null | string | undefined,
  ambientRequest: <R>(
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
    signal?: AbortSignal
  ) => Promise<R>,
  method: string,
  params: Record<string, unknown> = {},
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const owner = knownOwnerForSession(sessionId)

  try {
    assertSessionOwnerResolved(owner, { method, sessionId })
  } catch (error) {
    return Promise.reject(error)
  }

  return requestForSessionProfile<T>(owner, ambientRequest, method, params, timeoutMs, signal)
}

/** Resolve a session id THAT MAY BE A RUNTIME ID to the stored id its tile
 *  keys on. Session-scoped RPC params carry the runtime id, while tile owner
 *  routes (and everything else durable) key on the stored id — so routing an
 *  RPC by its own target session needs this translation first (#93080 /
 *  Bot Mode misroute). Ids that match a tile's stored id pass through, so
 *  callers can hand in either identity. Unknown ids return null: the caller
 *  falls back to its ambient routing rather than guessing. */
export function storedSessionIdForRuntimeId(sessionId: string): null | string {
  const tiles = $sessionTiles.get()

  // Stored-id claims are authoritative (durable identity): check them all
  // before any runtime binding, so a stale tile whose dead runtimeId collides
  // with a live tile's stored id cannot hijack the lookup.
  for (const tile of tiles) {
    if (tile.storedSessionId === sessionId) {
      return tile.storedSessionId
    }
  }

  for (const tile of tiles) {
    if (tile.runtimeId && tile.runtimeId === sessionId) {
      return tile.storedSessionId
    }
  }

  // The per-runtime state mirror carries the stored id the wiring cache bound
  // (ensureSessionState / a resume). This is how a MAIN-PANE runtime id — an
  // approval.respond from a native notification, a queued send — finds its
  // durable identity, and through it the exact owner (hint / tagged row).
  // Without this rung such ids fell straight to the ambient socket.
  const mirrored = $sessionStates.get()[sessionId]?.storedSessionId?.trim()

  return mirrored || null
}

export function setSessionTileWorkspaceScope(storedSessionId: string, scope: SessionTileWorkspaceScope): boolean {
  const tile = $sessionTiles.get().find(candidate => candidate.storedSessionId === storedSessionId)
  const workspaceOwnerKey = scope.workspaceMode === 'bots' ? scope.workspaceOwnerKey : undefined
  const ownerRoute = scope.workspaceMode === 'bots' ? scope.ownerRoute : undefined
  const workspaceTabTitle = scope.workspaceMode === 'bots' ? scope.workspaceTabTitle : undefined

  if (
    !tile ||
    ((tile.workspaceMode ?? 'sessions') === scope.workspaceMode &&
      tile.workspaceOwnerKey === workspaceOwnerKey &&
      tile.ownerRoute?.connectionId === ownerRoute?.connectionId &&
      tile.ownerRoute?.profile === ownerRoute?.profile &&
      tile.ownerRoute?.targetProfile === ownerRoute?.targetProfile &&
      tile.workspaceTabTitle === workspaceTabTitle)
  ) {
    return false
  }

  patchSessionTile(storedSessionId, {
    ownerRoute,
    workspaceMode: scope.workspaceMode,
    workspaceOwnerKey,
    workspaceTabTitle
  })

  return true
}

/** Drop live runtime bindings so every tile re-resumes — used on gateway
 *  reconnect, where a respawned backend re-mints (recycles) runtime ids.
 *  Also invalidates the wiring cache's stored→runtime map: clearing only the
 *  tile atoms left `resumeTile`'s warm path free to re-bind the same dead
 *  runtime id from the cache, so post-wake tiles repainted empty and never
 *  actually re-resumed. */
export interface RuntimeReconnectScope {
  connectionId: string
  profile?: null | string
}

/** Fallback scope for a restarted connection whose registry identity is
 *  unknown (a legacy remote primary with no connectionId). We cannot name the
 *  dead owner, so instead preserve only Bot runtimes whose owner is provably
 *  alive elsewhere; every other binding is dropped and re-resumes. A reset
 *  only costs a re-resume, so unknown owners fail toward recovery. */
export interface UnknownRuntimeReconnectScope {
  liveConnectionIds: ReadonlySet<string>
}

export function resetTileRuntimeBindings(
  reconnectedScope?: null | string | RuntimeReconnectScope | UnknownRuntimeReconnectScope
) {
  const tiles = $sessionTiles.get()

  const liveConnectionIds =
    reconnectedScope && typeof reconnectedScope === 'object' && 'liveConnectionIds' in reconnectedScope
      ? reconnectedScope.liveConnectionIds
      : null

  const reconnected =
    typeof reconnectedScope === 'string'
      ? { connectionId: reconnectedScope.trim(), profile: null }
      : reconnectedScope && !liveConnectionIds
        ? {
            connectionId: (reconnectedScope as RuntimeReconnectScope).connectionId.trim(),
            profile: (reconnectedScope as RuntimeReconnectScope).profile?.trim() || null
          }
        : null

  const belongsToReconnectedRuntime = (tile: SessionTile): boolean => {
    const route = tile.ownerRoute

    if (liveConnectionIds) {
      // Unknown restarted identity: a tile survives only when its owner is a
      // connection we know is still live — anything else rebinds on resume.
      return !route?.connectionId || !liveConnectionIds.has(route.connectionId)
    }

    if (!reconnected?.connectionId || route?.connectionId !== reconnected.connectionId) {
      return false
    }

    return !reconnected.profile || (route.targetProfile || route.profile) === reconnected.profile
  }

  const preservedStoredIds = new Set(
    tiles
      .filter(
        tile =>
          tile.workspaceMode === 'bots' &&
          Boolean(tile.ownerRoute?.connectionId) &&
          (!(reconnected || liveConnectionIds) || !belongsToReconnectedRuntime(tile))
      )
      .map(tile => tile.storedSessionId)
  )

  sessionTileDelegate()?.invalidateRuntimeBindings?.(preservedStoredIds)

  if (tiles.some(tile => tile.runtimeId && !preservedStoredIds.has(tile.storedSessionId))) {
    $sessionTiles.set(tiles.map(tile => (preservedStoredIds.has(tile.storedSessionId) ? tile : toStored(tile))))
  }
}

/** Unbind ONE reclaimed runtime from whichever tile holds it — the targeted
 *  sibling of resetTileRuntimeBindings. The reconnect-time reset can't cover a
 *  backend reclaim: the WS re-dials immediately, but the orphan reaper fires a
 *  grace window LATER, so the reclaim lands after every reconnect-path unbind
 *  already ran. Without this, the tile keeps pointing at the dead runtime whose
 *  state `session.reclaimed` just dropped — an empty transcript under live
 *  chrome — and SessionTilePane's resume effect (gated on `!runtimeId`) never
 *  re-resumes. Clearing the binding re-arms that effect, which rebinds a fresh
 *  runtime from the stored row. The pane itself stays: the stored session is
 *  intact, only its live runtime was reclaimed. */
export function unbindTileRuntime(runtimeId: string) {
  const tiles = $sessionTiles.get()

  if (tiles.some(t => t.runtimeId === runtimeId)) {
    $sessionTiles.set(tiles.map(t => (t.runtimeId === runtimeId ? { ...t, runtimeId: undefined } : t)))
  }
}

// ---------------------------------------------------------------------------
// Delegate — the wiring layer (which owns the gateway + session cache) plugs
// its actions in; tile UI calls through here. Same inversion as the tree
// store's pane closers.
// ---------------------------------------------------------------------------

export interface SessionTileDelegate {
  /** Archive a stored session (the sidebar's archive, incl. tile cleanup). */
  archiveSession(storedSessionId: string): Promise<void>
  /** Branch a stored session into a new chat (the sidebar's branch). */
  branchSession(storedSessionId: string): Promise<void>
  /** Delete a stored session (the sidebar's delete, incl. tile cleanup). */
  deleteSession(storedSessionId: string): Promise<void>
  /** Run a slash command against a tile's session (app-level effects — e.g.
   *  branch/handoff — act on the main surface, as they should). */
  executeSlash(rawCommand: string, sessionId: string): Promise<void>
  /** Interrupt a tile's running turn. */
  interruptSession(runtimeId: string): Promise<void>
  /** Drop the wiring cache's stored→runtime bindings. Called on gateway
   *  reconnect: a respawned backend re-mints runtime ids, so every binding
   *  recorded before the reconnect is suspect — without this, `resumeTile`'s
   *  warm path re-binds tiles to dead runtime ids (the sleep/wake "empty
   *  right pane" bug). Bindings re-record from live post-reconnect events. */
  invalidateRuntimeBindings?(preserveStoredSessionIds?: ReadonlySet<string>): void
  /** Bind a live runtime id for a stored session (resume without touching
   *  the main view). Returns the runtime id, or throws.
   *  `refreshTranscript` forces a REST merge even when a warm cached
   *  transcript already exists — reopen-after-idle must not paint the
   *  snapshot that was current when the panel last had a socket. */
  resumeTile(storedSessionId: string, options?: { refreshTranscript?: boolean }): Promise<string>
  /** Retire one runtime's busy/awaiting claim through the wiring cache
   *  (updateSessionState), so cache, focused view, busyRef, and tile mirrors
   *  settle together. Returns false when the cache holds no busy state for
   *  it — the caller downgrades the mirror itself. Reconnect-time twin of
   *  invalidateRuntimeBindings (#93059). */
  retireBusyClaim?(runtimeId: string): boolean
  /** Submit a prompt to a tile's live session. */
  submitToSession(runtimeId: string, text: string): Promise<void>
  /** THE session-state write path — routes through the wiring cache so the
   *  cache, the primary view (when active), and every tile mirror agree. */
  updateSession(runtimeId: string, updater: (state: ClientSessionState) => ClientSessionState): ClientSessionState
}

let delegate: SessionTileDelegate | null = null
export const $sessionTileDelegateRevision = atom(0)

export function setSessionTileDelegate(next: SessionTileDelegate) {
  delegate = next
  $sessionTileDelegateRevision.set($sessionTileDelegateRevision.get() + 1)
}

export function sessionTileDelegate(): SessionTileDelegate | null {
  return delegate
}

/** Reorder tiles to match layout-tree encounter order (stored ids in the order
 *  their `session-tile:` panes are walked). Restore replays the array through
 *  sequential adoption (each center tile APPENDS after the ones before it), so
 *  array order IS strip order — no `before` stamping needed; a stale `before`
 *  naming an absent pane falls back to append anyway (see insertAtGroup). Tiles
 *  not yet adopted sort after placed ones, stably. Returns `null` when nothing
 *  moves so callers can skip a needless persist. */
export function orderTilesByTree<T extends { storedSessionId: string }>(
  tree: LayoutNode | null,
  tiles: readonly T[]
): null | T[] {
  if (!tree || tiles.length < 2) {
    return null
  }

  const order: string[] = []

  const walk = (node: LayoutNode) => {
    if (node.type === 'group') {
      for (const id of node.panes) {
        if (id.startsWith(TILE_PANE_PREFIX)) {
          order.push(id.slice(TILE_PANE_PREFIX.length))
        }
      }

      return
    }

    node.children.forEach(walk)
  }

  walk(tree)

  const rank = new Map(order.map((id, i) => [id, i]))

  const next = [...tiles].sort(
    (a, b) => (rank.get(a.storedSessionId) ?? Infinity) - (rank.get(b.storedSessionId) ?? Infinity)
  )

  return next.some((t, i) => t !== tiles[i]) ? next : null
}

function syncTileStripOrder() {
  const next = orderTilesByTree($layoutTree.get(), $sessionTiles.get())

  if (next) {
    saveTiles(next)
  }
}

/** Open a tile for a stored session, or MOVE an existing one to the new dock
 *  (`dir`; `center` = stack into the anchor's zone, `before` = strip slot). The
 *  move path is what lets a tile's own TAB be dragged like a sidebar row — drop
 *  it on a zone/edge/strip and the tile goes there (drop-on-a-composer links
 *  instead, handled by the drag resolver). The session LOADED IN MAIN never
 *  opens as a tile (same transcript twice, fighting one runtime — silly).
 *
 *  An unanchored open (⌘T, ⌘⇧T on a tile that predates anchors) docks into the
 *  FOCUSED chat zone — the same zone ⌘1…⌘9 and ⌘W act on — so a new tab lands
 *  in the strip the user is looking at, not always main's. */
export function openSessionTile(
  storedSessionId: string,
  dir: TileDock = 'right',
  anchor?: string,
  before?: null | string,
  workspaceScope: SessionTileWorkspaceScope = { workspaceMode: 'sessions' }
) {
  const tiles = $sessionTiles.get()

  // Opening a session in a tab/tile is "reading" it — clear its unread dot
  // exactly like main-thread resume does. Previously only
  // setSelectedStoredSessionId cleared unread, so tile-opened sessions kept
  // their green dot even while the user was reading them. Acks the persisted
  // watermark/marker too so a later list refresh doesn't repaint it.
  markSessionRead(storedSessionId)
  ackStoredSessionId(storedSessionId)

  if (workspaceScope.workspaceMode === 'sessions' && storedSessionId === $selectedStoredSessionId.get()) {
    return
  }

  const dock = anchor ?? focusedSessionTabAnchor() ?? undefined

  const workspaceOwnerKey = workspaceScope.workspaceMode === 'bots' ? workspaceScope.workspaceOwnerKey : undefined

  if (!tiles.some(t => t.storedSessionId === storedSessionId)) {
    saveTiles([
      ...tiles,
      {
        anchor: dock,
        before,
        dir,
        ownerRoute: workspaceScope.workspaceMode === 'bots' ? workspaceScope.ownerRoute : undefined,
        storedSessionId,
        workspaceMode: workspaceScope.workspaceMode,
        workspaceOwnerKey,
        workspaceTabTitle: workspaceScope.workspaceMode === 'bots' ? workspaceScope.workspaceTabTitle : undefined
      }
    ])
    // Adoption is async via the registry — order sync runs after the move path
    // below; a brand-new tile's strip slot is already in `before`.

    return
  }

  setSessionTileWorkspaceScope(storedSessionId, workspaceScope)

  // Already open: relocate the existing pane to the drop target (pane-mirror
  // only docks on first adoption, so a re-drag must move the tree pane itself).
  const tree = $layoutTree.get()
  const target = tree ? findGroupOfPane(tree, dock ?? 'workspace')?.id : null

  if (target) {
    moveTreePane(`${TILE_PANE_PREFIX}${storedSessionId}`, { before: before ?? null, groupId: target, pos: dir })
    patchSessionTile(storedSessionId, { anchor: dock, before: before ?? undefined, dir })
    syncTileStripOrder()
  }
}

/** ⌘W on the MAIN tab: the next session tab stacked WITH the workspace, to
 *  shift into main. Walks the workspace group's strip from the workspace tab
 *  outward (the tab after it first, then wrapping to the ones before), and
 *  returns the first session tile's stored id. Null when the workspace has no
 *  session tab stacked beside it (⌘W then stays the no-op it was). */
export function nextSessionTileForWorkspace(): null | string {
  const tree = $layoutTree.get()
  const group = tree ? findGroupOfPane(tree, 'workspace') : null

  if (!group) {
    return null
  }

  const tiles = $sessionTiles.get()
  const idx = group.panes.indexOf('workspace')
  // After the workspace tab first, then the ones before it (nearest-out).
  const ordered = [...group.panes.slice(idx + 1), ...group.panes.slice(0, idx).reverse()]

  for (const paneId of ordered) {
    if (paneId.startsWith(TILE_PANE_PREFIX)) {
      const storedSessionId = paneId.slice(TILE_PANE_PREFIX.length)

      if (tiles.some(t => t.storedSessionId === storedSessionId)) {
        return storedSessionId
      }
    }
  }

  // Nothing stacked WITH main — but a session tile in another zone can still
  // shift in. Without this, closing main in a side-by-side layout skipped
  // promotion entirely and dropped to a fresh "New session" draft, which read
  // as "closing a pane gave me a new session" (#88924). Promoting the tile
  // also collapses its zone, so Close is how a multi-pane layout shrinks.
  for (const tile of tiles) {
    if (tree && findGroupOfPane(tree, `${TILE_PANE_PREFIX}${tile.storedSessionId}`)) {
      return tile.storedSessionId
    }
  }

  return null
}

/** If a session is already ON SCREEN — an open tile OR the one loaded in main —
 *  front its tab (and focus its zone) and report WHICH. A sidebar click on an
 *  already-open chat JUMPS to its tab instead of reloading it; `null` means the
 *  caller must load it into main. Covers the two dead clicks: an open tile, and
 *  the main session while focus sits on a tile (route unchanged → no reload).
 *  Callers that own the router need the `'main'` vs `'tile'` distinction: a
 *  `'main'` hit only reaches the screen if the workspace pane is actually
 *  showing the chat, whereas a tile renders in its own pane regardless. */
export function focusOpenSession(
  storedSessionId: string,
  workspaceScope: SessionTileWorkspaceScope = { workspaceMode: 'sessions' }
): 'main' | 'tile' | null {
  if ($sessionTiles.get().some(t => t.storedSessionId === storedSessionId)) {
    const paneId = `${TILE_PANE_PREFIX}${storedSessionId}`
    revealTreePane(paneId) // un-dismiss + adopt + front in its group
    const tree = $layoutTree.get()
    const group = tree ? findGroupOfPane(tree, paneId) : null

    if (group) {
      noteActiveTreeGroup(group.id)
    }

    return 'tile'
  }

  // Already the main session: front the workspace tab and drop tile focus so
  // the readouts + sidebar highlight come home (a no-op when main is focused).
  if (workspaceScope.workspaceMode === 'sessions' && storedSessionId === $selectedStoredSessionId.get()) {
    revealTreePane('workspace')
    noteActiveTreeGroup(null)

    return 'main'
  }

  return null
}

/** Front the tab a Bot Mode owner already has open and report its stored id:
 *  the tile the zone last had active for `workspaceOwnerKey` (the same
 *  window-local memory the strip restores on a scope switch), else the most
 *  recently opened one. `null` when that owner has no open tile — the caller
 *  decides what to open then. A roster click consults this FIRST so a bot
 *  with open tabs comes back to the one the user left, instead of re-opening
 *  its canonical Bot Chat beside them: nothing records a tab close except the
 *  tile bucket forgetting it, so any open path that ignores the open set
 *  resurrects closed chats on every bot switch. */
export function focusWorkspaceOwnerSessionTile(workspaceOwnerKey: string): null | string {
  const owned = $sessionTiles
    .get()
    .filter(tile => tile.workspaceMode === 'bots' && tile.workspaceOwnerKey === workspaceOwnerKey)

  if (owned.length === 0) {
    return null
  }

  // Most recent first, so the fallback (no remembered pane) is the newest tab.
  const paneIds = owned.map(tile => `${TILE_PANE_PREFIX}${tile.storedSessionId}`).reverse()
  const paneId = resolveRememberedActivePane(workspaceScopeKey('bots', workspaceOwnerKey), paneIds) ?? paneIds[0]
  const storedSessionId = paneId.slice(TILE_PANE_PREFIX.length)

  focusOpenSession(storedSessionId, { workspaceMode: 'bots', workspaceOwnerKey })

  return storedSessionId
}

/** Does a sidebar click still need to navigate after `focusOpenSession`? A miss
 *  always does. A `'main'` hit does too while the workspace pane is showing a
 *  full page (artifacts, skills, …): fronting the workspace tab doesn't put the
 *  chat back on screen — only a route change back to the session does. A tile
 *  hit never does; its pane renders the chat regardless of the route. */
export function focusedSessionNeedsRoute(focused: 'main' | 'tile' | null, workspaceIsPage: boolean): boolean {
  return !focused || (focused === 'main' && workspaceIsPage)
}

/** The open tab that's still an empty "New session" draft, if there is one.
 *  That tab is the one the user would have typed into, so an open-from-nowhere
 *  spends it instead of stacking a second blank tab beside it. Most recent
 *  wins; a tile whose runtime hasn't bound (or whose state hasn't published) is
 *  unknown rather than empty, so it's left alone. */
export function blankDraftTile(
  tiles: readonly SessionTile[],
  states: Record<string, ClientSessionState>
): null | SessionTile {
  return (
    tiles.findLast(({ runtimeId }) => {
      const state = runtimeId ? states[runtimeId] : undefined

      return Boolean(state && !state.busy && state.messages.length === 0)
    }) ?? null
  )
}

/** Hand an open blank draft tab over to `storedSessionId`, keeping its slot.
 *  False when there's no such tab, so the caller can fall back. The spent draft
 *  is DISCARDED rather than closed: it never held a conversation, so ⌘⇧T
 *  resurrecting it would just restore an empty tab. */
export function reuseBlankDraftTile(
  storedSessionId: string,
  workspaceScope: SessionTileWorkspaceScope = { workspaceMode: 'sessions' }
): boolean {
  const tile = blankDraftTile($sessionTiles.get(), $sessionStates.get())

  if (!tile || tile.storedSessionId === storedSessionId) {
    return false
  }

  discardSessionTile(tile.storedSessionId)
  openSessionTile(storedSessionId, tile.dir, tile.anchor, tile.before, workspaceScope)
  revealTreePane(`${TILE_PANE_PREFIX}${storedSessionId}`)

  return true
}

// Closed-tab stack for ⌘⇧T reopen (in-memory) — keyed PER PROFILE like the
// tiles themselves, so ⌘⇧T after a profile switch never resurrects the other
// profile's session. The tile's placement is remembered so it returns in place.
const closedTilesByProfile: Record<string, SessionTile[]> = {}
const closedStack = (): SessionTile[] => (closedTilesByProfile[profileKey()] ??= [])

export function closeSessionTile(storedSessionId: string) {
  const tile = $sessionTiles.get().find(t => t.storedSessionId === storedSessionId)

  if (tile) {
    closedStack().push(toStored(tile))
  }

  saveTiles($sessionTiles.get().filter(t => t.storedSessionId !== storedSessionId))

  // A settled session may never publish again, so the publish-time eviction
  // in publishSessionState can't reach it — drop its cached state here. A
  // BUSY one stays: its turn keeps streaming in the background, the sidebar
  // dot reads it, and settle evicts it. ⌘⇧T reopen re-publishes from the
  // wiring cache (resumeTile's warm path), so nothing is lost.
  const runtimeId = tile?.runtimeId
  const state = runtimeId ? $sessionStates.get()[runtimeId] : undefined

  if (runtimeId && state && evictable(runtimeId, state)) {
    dropSessionState(runtimeId)
  }
}

/** Persist-close every session tile whose pane lives in `paneId`'s group.
 *
 * Close All used to only dismiss layout-tree panes. Bot Mode tiles are
 * stored in the shared `__bots_workspace__` bucket, so a later roster
 * click or profile swap rehydrated `$sessionTiles` and the closed tabs
 * came back (#94137). Routing through {@link closeSessionTile} writes that
 * bucket, so the closed set survives those rehydrations and a restart.
 */
export function closeAllOpenSessionTiles(paneId: string): void {
  const tree = $layoutTree.get()
  // Copy the live group list. closeSessionTile can rewrite the layout
  // tree; iterating the original array would skip every other pane.
  const panes = [...((tree ? findGroupOfPane(tree, paneId) : null)?.panes ?? [])]

  for (const id of panes) {
    if (id.startsWith(TILE_PANE_PREFIX)) {
      closeSessionTile(id.slice(TILE_PANE_PREFIX.length))
    }
  }
}

/** Drop a DEAD tile — a persisted tile whose session no longer exists on the
 *  backend (resume 404s). Unlike close, it leaves no ⌘⇧T undo (resurrecting it
 *  would just 404 again) and evicts any cached state. This is what clears the
 *  "Session not found" resume spam from stale/cross-profile persisted tiles. */
export function discardSessionTile(storedSessionId: string) {
  const runtimeId = $sessionTiles.get().find(t => t.storedSessionId === storedSessionId)?.runtimeId

  if (runtimeId) {
    dropSessionState(runtimeId)
  }

  saveTiles($sessionTiles.get().filter(t => t.storedSessionId !== storedSessionId))
}

/**
 * Drop every persisted tile owned by a profile that is being deleted — the
 * profile's own session-tile bucket and any Bot Mode tile whose ownerRoute
 * points at it (matched by desktop profile name, or by exact connection /
 * backend target profile when a source-scoped route is given).
 *
 * A leftover tile RESURRECTS the deleted profile on the next launch: Bot tab
 * restore re-dials the profile's backend, whose ensure_hermes_home() re-creates
 * the profile directory the delete just removed (hermes-agent#94235). Same
 * discard (no ⌘⇧T) semantics as discardSessionTile — undoing the delete of the
 * owning profile would resolve to a 404 again.
 */
export function dropTilesForProfile(
  profile: string,
  route?: { connectionId?: string; profile?: string; targetProfile?: string }
): void {
  const name = normalizeProfileKey(profile)
  // Route fields go through the SAME canonicalization as `name` below — a
  // source-scoped delete must not be defeated by stray whitespace around a
  // profile name that a non-route delete trims away.
  const routeProfile = route?.profile ? normalizeProfileKey(route.profile) : ''
  const routeTarget = route?.targetProfile ? normalizeProfileKey(route.targetProfile) : ''
  const routeConnection = String(route?.connectionId ?? '').trim()

  const ownerMatches = (owner: SessionProfileRoute | undefined): boolean => {
    if (!owner) {
      return false
    }

    const ownerProfile = normalizeProfileKey(owner.profile)
    const ownerTarget = normalizeProfileKey(owner.targetProfile)
    const ownerConnection = String(owner.connectionId ?? '').trim()

    if (routeProfile) {
      // Source-scoped delete: the route's desktop profile name, backend target,
      // and connection must all agree with the tile's owner route.
      if (ownerProfile !== routeProfile) {
        return false
      }

      if (routeTarget && ownerTarget !== routeTarget) {
        return false
      }

      return !routeConnection || ownerConnection === routeConnection
    }

    // Desktop-local delete: also require the tile's owner connection to be the
    // LOCAL connection. A same-named bot on another connection is a different
    // agent — the deleted local profile never owned it, and dropping its tile
    // would orphan a live conversation (hermes-agent#94235).
    return (ownerProfile === name || ownerTarget === name) && ownerConnection === 'local'
  }

  // The profile's own sessions bucket (Bot tiles live in the shared bucket
  // and are keyed by ownerRoute, not by bucket).
  delete tilesByProfile[name]

  const botTiles = tilesByProfile[BOTS_TILE_BUCKET]

  if (botTiles) {
    const remaining = botTiles.filter(tile => !ownerMatches(tile.ownerRoute))

    if (remaining.length > 0) {
      tilesByProfile[BOTS_TILE_BUCKET] = remaining
    } else {
      delete tilesByProfile[BOTS_TILE_BUCKET]
    }
  }

  // Live atom: drop the deleted profile's Bot tiles, and — when the deleted
  // profile IS the live gateway's profile — the session tiles in view (they
  // belong to that bucket; the caller re-homes afterwards).
  const live = $sessionTiles.get()

  const next = live.filter(tile =>
    tile.workspaceMode === 'bots' ? !ownerMatches(tile.ownerRoute) : profileKey() !== name
  )

  if (next.length !== live.length) {
    $sessionTiles.set(next)
  }

  persistTiles()
}

/** ⌘⇧T — reopen the most recently closed tab where it was, then focus it.
 *  Adoption alone is silent (won't steal the active tab), so restore has to
 *  front the pane explicitly. Skips ids that are live again (reopened / now
 *  the primary). */
export function reopenLastClosedTile(): void {
  const stack = closedStack()

  for (let tile = stack.pop(); tile; tile = stack.pop()) {
    const { storedSessionId } = tile

    if (storedSessionId === $selectedStoredSessionId.get()) {
      continue
    }

    if (!$sessionTiles.get().some(t => t.storedSessionId === storedSessionId)) {
      openSessionTile(storedSessionId, tile.dir, tile.anchor, tile.before, {
        workspaceMode: tile.workspaceMode ?? 'sessions',
        workspaceOwnerKey: tile.workspaceOwnerKey
      })
      focusOpenSession(storedSessionId)

      return
    }
  }
}

// ---------------------------------------------------------------------------
// The FOCUSED session — one derivation, not another hand-maintained
// "$activeSession" sibling. The layout's interaction tracker ($activeTreeGroup:
// last click/focus, the same source ⌘W uses) resolves to a zone; its active
// pane names the session: a `session-tile:<storedId>` pane IS that session,
// anything else falls back to the route-driven primary. Chrome that should
// follow the user between tiles (titlebar session title, statusbar context /
// timer / model) reads these instead of the primary-only atoms.
// ---------------------------------------------------------------------------

/** Stored id of the focused session (the interacted zone's tile, else the
 *  primary's selection). Null on a fresh draft. */
export const $focusedStoredSessionId = computed(
  [$activeTreeGroup, $layoutTree, $selectedStoredSessionId, $workspaceMode],
  (groupId, tree, selected, workspaceMode) => {
    const active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

    if (active?.startsWith(TILE_PANE_PREFIX)) {
      return active.slice(TILE_PANE_PREFIX.length)
    }

    // The interaction tracker can point at sidebar CHROME while a chat still
    // holds the main zone's active tab — clicking a Bots-pane roster row moves
    // it to the sidebar group, whose active pane ('hermes-bots:pane') is not a
    // session tile. In sessions mode the primary selection answers, exactly as
    // always. In Bot Mode that fallback alone publishes a NULL "focused"
    // edge: bot chats open as TILES and never set $selectedStoredSessionId,
    // so the selection is null while the chat is plainly on screen. The Bots
    // plugin reads that null edge as "the chat lost the center", releases its
    // open claim, and re-asserts the Bots home over the still-visible chat —
    // the reported "clicking a bot chat jumps to the list" (#96062). Bot
    // Mode's on-screen truth is the main zone's active TILE; only when the
    // main zone holds no tile (chat closed) does the selection answer, so a
    // genuine close still lets the home return.
    if (workspaceMode === 'bots' && tree) {
      const mainActive = findGroupOfPane(tree, 'workspace')?.active

      if (mainActive?.startsWith(TILE_PANE_PREFIX)) {
        return mainActive.slice(TILE_PANE_PREFIX.length)
      }
    }

    return selected
  }
)

/** Every session currently OPEN as a surface: the primary's selection plus
 *  every tile's stored id. The sidebar highlights all of them (the focused one
 *  at full strength, the rest dimmed) so a multi-pane workspace shows which
 *  chats are on screen, not just the one being typed into. */
export const $openStoredSessionIds = computed(
  [$selectedStoredSessionId, $sessionTiles],
  (selected, tiles) => new Set([...(selected ? [selected] : []), ...tiles.map(t => t.storedSessionId)])
)

/** Live runtime id of the focused session (a tile's bound runtime, else the
 *  primary's active session). */
export const $focusedRuntimeId = computed(
  [$focusedStoredSessionId, $selectedStoredSessionId, $activeSessionId, $sessionTiles],
  (focused, selected, primaryRuntime, tiles) => {
    if (focused && focused !== selected) {
      return tiles.find(t => t.storedSessionId === focused)?.runtimeId ?? null
    }

    return primaryRuntime
  }
)

/** The focused session's state slice (undefined while unresolved/unbound). */
export const $focusedSessionState = computed([$focusedRuntimeId, $sessionStates], (runtimeId, states) =>
  runtimeId ? states[runtimeId] : undefined
)

/** A PRIMARY navigation (sidebar resume, route change, new chat) homes focus to
 *  the workspace — UNLESS the selected id is already an open TILE, where
 *  `focusOpenSession` owns the move and homing would yank every stacked tile
 *  behind the workspace (A+B "disappear" when switching to C). */
export const selectionHomesToWorkspace = (selected: null | string, tiles: readonly SessionTile[]): boolean =>
  !(selected && tiles.some(t => t.storedSessionId === selected))

// Bringing a finished session to the front clears its green dot. Keyed on the
// FOCUSED session, not the selected one: a tile is never $selectedStoredSessionId,
// and a tile tab click goes through activateTreePane rather than focusOpenSession,
// so this is the one hook that catches every way a tile reaches the front.
// Clears the whole conversation family (markSessionRead) AND acks the
// persisted watermark/marker (ackStoredSessionId) so the next list refresh
// doesn't repaint the dot the user just cleared by looking at it.
$focusedStoredSessionId.listen(focused => {
  if (focused) {
    markSessionRead(focused)
    ackStoredSessionId(focused)
  }
})

// Cold-start restore is the one selection change that is NOT a navigation: the
// route already pointed at the primary session before the window loaded, and
// homing on it would front the workspace tab over the PERSISTED active tab —
// then persist that clobber, so the tab you reloaded on never comes back
// (⌘R always landing on main). use-route-resume arms this one-shot right
// before dispatching the boot resume; the very next selection change skips
// homing and the restored layout tree keeps its say.
let selectionRestoreInFlight = false

export function markSelectionRestore() {
  selectionRestoreInFlight = true
}

// Homing also FRONTS the workspace tab: the resumed chat loads in the workspace
// pane, so a zone parked on a tile tab must switch back or the click looks dead.
$selectedStoredSessionId.listen(selected => {
  const restoring = selectionRestoreInFlight
  selectionRestoreInFlight = false

  if (restoring || !selectionHomesToWorkspace(selected, $sessionTiles.get())) {
    return
  }

  noteActiveTreeGroup(null)
  revealTreePane('workspace')
})

// Dev hook for automation (mirrors __HERMES_LAYOUT_TREE__).
if ((import.meta.env.DEV || import.meta.env.VITE_PERF_PROBE === '1') && typeof window !== 'undefined') {
  ;(window as unknown as Record<string, unknown>).__HERMES_SESSION_TILES__ = {
    close: closeSessionTile,
    drop: dropSessionState,
    open: openSessionTile,
    patch: patchSessionTile,
    publish: publishSessionState,
    /** Seed the recents list — models a populated sessions DB in perf runs. */
    seedSessions: (rows: SessionInfo[]) => setSessions(rows),
    sessions: () => $sessions.get(),
    states: () => $sessionStates.get(),
    tiles: () => $sessionTiles.get(),
    /** THE real gateway write path (wiring cache + journal + publish + view
     *  sync), unlike `publish` which only touches the store. Perf scenarios
     *  must drive this or they under-model streaming cost. */
    update: (runtimeId: string, updater: (state: ClientSessionState) => ClientSessionState) =>
      sessionTileDelegate()?.updateSession(runtimeId, updater)
  }
}
