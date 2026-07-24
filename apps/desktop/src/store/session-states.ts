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

import { atom, computed } from 'nanostores'

import type { ClientSessionState } from '@/app/types'
import { findGroup, findGroupOfPane, type LayoutNode } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $layoutTree,
  moveTreePane,
  noteActiveTreeGroup,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import { stableArray } from '@/lib/stable-array'
import { readJson, writeJson } from '@/lib/storage'

import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $unreadFinishedSessionIds,
  setActiveSessionStoredIdRotation
} from './session'
import { isSecondaryWindow } from './windows'

// ---------------------------------------------------------------------------
// Reactive per-runtime session state (view mirror of the wiring cache).
// ---------------------------------------------------------------------------

export const $sessionStates = atom<Record<string, ClientSessionState>>({})

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

// --- Watchdog: marks busy sessions quiet after 8 min of stream silence -----
export const SESSION_WATCHDOG_TIMEOUT_MS = 8 * 60 * 1000
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
  } else if (!next.busy && wasWorking) {
    markSettled(storedId)

    if (storedId !== $selectedStoredSessionId.get()) {
      const cur = $unreadFinishedSessionIds.get()

      if (!cur.includes(storedId)) {
        $unreadFinishedSessionIds.set([...cur, storedId])
      }
    }
  }
}

/** Publish one session's state. Automatically fires transition side-effects
 *  (watchdog arm/disarm, settle grace, unread marker, compression id rotation)
 *  by diffing previous vs next — callers never need to manually call a
 *  transition handler. */
export function publishSessionState(runtimeId: string, state: ClientSessionState) {
  const prev = $sessionStates.get()[runtimeId] ?? null
  $sessionStates.set({ ...$sessionStates.get(), [runtimeId]: state })
  handleTransition(prev, state, runtimeId)
}

export function dropSessionState(runtimeId: string) {
  // Disarm the watchdog — a dropped runtime must not fire a stale clear later.
  // Settle-grace entries are keyed by stored id and self-expire; leave them so
  // a just-finished session's row survives merge eviction even if its tile or
  // cached runtime is dropped in the meantime.
  clearWatchdog(runtimeId)

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
  $stalledSessionIds.set([])
  $sessionStates.set({})
}

// Derived per-session status sets — pure projections of `$sessionStates` (which
// holds `busy`/`needsInput` per runtime), keeping the data flow one-directional:
// gateway event → cache → $sessionStates → computed views.
//
// Perf: `$sessionStates` is republished on EVERY message delta (tens/sec during
// a turn), but these sets only change on busy/needsInput edges. `stableArray`
// keeps the prior reference when membership is unchanged so `computed` skips the
// emit — otherwise the whole sidebar + every row re-renders per token.
const storedIds = (states: Record<string, ClientSessionState>, pred: (s: ClientSessionState) => boolean) =>
  Object.values(states)
    .filter(s => pred(s) && s.storedSessionId)
    .map(s => s.storedSessionId!)

let workingIds: readonly string[] = []
export const $workingSessionIds = computed(
  $sessionStates,
  states =>
    (workingIds = stableArray(
      workingIds,
      storedIds(states, s => s.busy)
    ))
)

let attentionIds: readonly string[] = []
export const $attentionSessionIds = computed(
  $sessionStates,
  states =>
    (attentionIds = stableArray(
      attentionIds,
      storedIds(states, s => s.needsInput)
    ))
)

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

/** Persisted placement — `dir` + strip slot (`before`) + dock `anchor` so a
 *  restart / profile swap re-adopts tiles in the same order, not all stacked
 *  right of workspace. */
type StoredTile = Pick<SessionTile, 'anchor' | 'before' | 'dir' | 'storedSessionId'>

const toStored = (t: SessionTile): StoredTile => ({
  anchor: t.anchor,
  before: t.before,
  dir: t.dir,
  storedSessionId: t.storedSessionId
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
            storedSessionId: raw.storedSessionId
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

      if (tiles.length > 0) {
        byProfile[normalizeProfileKey(profile)] = tiles
      }
    }
  }

  // Migrate a v1 flat list into the default profile, then retire the key.
  const legacy = parseTileList(readJson<unknown>(LEGACY_TILES_KEY))

  if (legacy.length > 0) {
    const key = normalizeProfileKey('default')
    byProfile[key] = [...(byProfile[key] ?? []), ...legacy]
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
export const $sessionTiles = atom<SessionTile[]>(isSecondaryWindow() ? [] : [...(tilesByProfile[profileKey()] ?? [])])

function persistTiles() {
  // Shares the origin's storage; a secondary window holds no tiles, so a write
  // back would only wipe the primary's set.
  if (isSecondaryWindow()) {
    return
  }

  writeJson(TILES_KEY, Object.keys(tilesByProfile).length === 0 ? null : tilesByProfile)
}

function saveTiles(tiles: SessionTile[]) {
  $sessionTiles.set(tiles)
  const stored = tiles.map(toStored)

  if (stored.length > 0) {
    tilesByProfile[profileKey()] = stored
  } else {
    delete tilesByProfile[profileKey()]
  }

  persistTiles()
}

// Profile switch: surface the new profile's tiles with runtime ids cleared so
// they re-resume against the now-current gateway. (Fires immediately on
// subscribe; harmless — the init value already matches.) A secondary window
// never carries tiles, so it stays out of this entirely.
if (!isSecondaryWindow()) {
  $activeGatewayProfile.subscribe(() => {
    $sessionTiles.set([...(tilesByProfile[profileKey()] ?? [])])
  })
}

export function patchSessionTile(storedSessionId: string, patch: Partial<SessionTile>) {
  saveTiles($sessionTiles.get().map(t => (t.storedSessionId === storedSessionId ? { ...t, ...patch } : t)))
}

/** Drop live runtime bindings so every tile re-resumes — used on gateway
 *  reconnect, where a respawned backend re-mints (recycles) runtime ids. */
export function resetTileRuntimeBindings() {
  const tiles = $sessionTiles.get()

  if (tiles.some(t => t.runtimeId)) {
    $sessionTiles.set(tiles.map(toStored))
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
  /** Bind a live runtime id for a stored session (resume without touching
   *  the main view). Returns the runtime id, or throws. */
  resumeTile(storedSessionId: string): Promise<string>
  /** Submit a prompt to a tile's live session. */
  submitToSession(runtimeId: string, text: string): Promise<void>
  /** THE session-state write path — routes through the wiring cache so the
   *  cache, the primary view (when active), and every tile mirror agree. */
  updateSession(runtimeId: string, updater: (state: ClientSessionState) => ClientSessionState): ClientSessionState
}

let delegate: SessionTileDelegate | null = null

export function setSessionTileDelegate(next: SessionTileDelegate) {
  delegate = next
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
 *  opens as a tile (same transcript twice, fighting one runtime — silly). */
export function openSessionTile(
  storedSessionId: string,
  dir: TileDock = 'right',
  anchor?: string,
  before?: null | string
) {
  const tiles = $sessionTiles.get()

  if (storedSessionId === $selectedStoredSessionId.get()) {
    return
  }

  if (!tiles.some(t => t.storedSessionId === storedSessionId)) {
    saveTiles([...tiles, { anchor, before, dir, storedSessionId }])
    // Adoption is async via the registry — order sync runs after the move path
    // below; a brand-new tile's strip slot is already in `before`.

    return
  }

  // Already open: relocate the existing pane to the drop target (pane-mirror
  // only docks on first adoption, so a re-drag must move the tree pane itself).
  const tree = $layoutTree.get()
  const target = tree ? findGroupOfPane(tree, anchor ?? 'workspace')?.id : null

  if (target) {
    moveTreePane(`${TILE_PANE_PREFIX}${storedSessionId}`, { before: before ?? null, groupId: target, pos: dir })
    patchSessionTile(storedSessionId, { anchor, before: before ?? undefined, dir })
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

  return null
}

/** If a session is already ON SCREEN — an open tile OR the one loaded in main —
 *  front its tab (and focus its zone) and return true. A sidebar click on an
 *  already-open chat JUMPS to its tab instead of reloading it; `false` means the
 *  caller must load it into main. Covers the two dead clicks: an open tile, and
 *  the main session while focus sits on a tile (route unchanged → no reload). */
export function focusOpenSession(storedSessionId: string): boolean {
  if ($sessionTiles.get().some(t => t.storedSessionId === storedSessionId)) {
    const paneId = `${TILE_PANE_PREFIX}${storedSessionId}`
    revealTreePane(paneId) // un-dismiss + adopt + front in its group
    const tree = $layoutTree.get()
    const group = tree ? findGroupOfPane(tree, paneId) : null

    if (group) {
      noteActiveTreeGroup(group.id)
    }

    return true
  }

  // Already the main session: front the workspace tab and drop tile focus so
  // the readouts + sidebar highlight come home (a no-op when main is focused).
  if (storedSessionId === $selectedStoredSessionId.get()) {
    revealTreePane('workspace')
    noteActiveTreeGroup(null)

    return true
  }

  return false
}

// Closed-tab stack for ⌘⇧T reopen (in-memory) — keyed PER PROFILE like the
// tiles themselves, so ⌘⇧T after a profile switch never resurrects the other
// profile's session. The tile's placement is remembered so it returns in place.
const closedTilesByProfile: Record<string, SessionTile[]> = {}
const closedStack = (): SessionTile[] => (closedTilesByProfile[profileKey()] ??= [])

export function closeSessionTile(storedSessionId: string) {
  const tile = $sessionTiles.get().find(t => t.storedSessionId === storedSessionId)

  if (tile) {
    closedStack().push({ anchor: tile.anchor, before: tile.before, dir: tile.dir, storedSessionId })
  }

  saveTiles($sessionTiles.get().filter(t => t.storedSessionId !== storedSessionId))
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

/** ⌘⇧T — reopen the most recently closed tab where it was. Skips ids that are
 *  live again (reopened, or now the primary). */
export function reopenLastClosedTile(): void {
  const stack = closedStack()

  for (let tile = stack.pop(); tile; tile = stack.pop()) {
    const { storedSessionId } = tile

    if (storedSessionId === $selectedStoredSessionId.get()) {
      continue
    }

    if (!$sessionTiles.get().some(t => t.storedSessionId === storedSessionId)) {
      openSessionTile(storedSessionId, tile.dir, tile.anchor, tile.before)

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
  [$activeTreeGroup, $layoutTree, $selectedStoredSessionId],
  (groupId, tree, selected) => {
    const active = groupId && tree ? findGroup(tree, groupId)?.active : undefined

    return active?.startsWith(TILE_PANE_PREFIX) ? active.slice(TILE_PANE_PREFIX.length) : selected
  }
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

// Homing also FRONTS the workspace tab: the resumed chat loads in the workspace
// pane, so a zone parked on a tile tab must switch back or the click looks dead.
$selectedStoredSessionId.listen(selected => {
  if (!selectionHomesToWorkspace(selected, $sessionTiles.get())) {
    return
  }

  noteActiveTreeGroup(null)
  revealTreePane('workspace')
})

// Dev hook for automation (mirrors __HERMES_LAYOUT_TREE__).
if (import.meta.env.DEV && typeof window !== 'undefined') {
  ;(window as unknown as Record<string, unknown>).__HERMES_SESSION_TILES__ = {
    close: closeSessionTile,
    open: openSessionTile,
    patch: patchSessionTile,
    publish: publishSessionState,
    states: () => $sessionStates.get(),
    tiles: () => $sessionTiles.get()
  }
}
