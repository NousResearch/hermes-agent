import { atom, computed } from 'nanostores'

import { normalizeSessionSource, SIDEBAR_EXCLUDED_SOURCES } from '@/lib/session-source'
import { normalizeProfileKey } from '@/store/profile'
import { $cronSessions, $messagingSessions, $sessions, $unlistedSessionOwnerRows, sessionMatchesStoredId } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

/**
 * Rows for sessions that are LIVE in the gateway process but have no persisted
 * row yet (`session.active_list` entries absent from every DB-backed sidebar
 * slice). A session created over the gateway by another client — TUI, CLI, a
 * second window — is invisible to the stored-list datasource until its first
 * prompt persists a DB row (the sidebar slice filters `min_message_count=1`),
 * and the renderer only inserts optimistic rows for creates IT performed
 * (#50799). This atom is the dedicated live group that closes that gap: the
 * sidebar renders it beside the stored rows, and promotion is automatic —
 * once the DB row lands and the list refresh returns it, the dedupe below
 * drops the live row.
 *
 * Mirrors the TUI switcher's `[new][live…][history…]` split (its
 * `resumableHistory()` dedupes live against resumable rows the same way).
 *
 * Lifecycle: `reconcileLiveSessions()` is fed by the existing `session.active_list`
 * poll (use-background-sync); a connection/profile switch must call
 * `clearLiveSessions()` alongside the other gateway-bound store wipes
 * (see store/gateway-switch.ts).
 *
 * One duplicate the dedupe cannot catch inside its own poll cycle:
 * auto-compression rotates a live session's `session_key` to the new tip while
 * Recents still projects the OLD tip, whose lineage does not contain the new
 * key — for up to the stored refresh's trailing gap (`SESSIONS_LIST_TICK_GAP_MS`)
 * the session can show in both lists. It self-heals on the next list refresh and
 * this group's rows are read-only, so the cost is cosmetic.
 */
export const $liveSessions = atom<SessionInfo[]>([])

/** One `session.active_list` entry as the desktop consumes it. The current
 *  backend contract (tui_gateway SessionActiveItem) guarantees
 *  id/last_active/message_count/model/preview/session_key/started_at/status/
 *  title; `source`, `hidden`, and `profile` are additive fields older
 *  backends simply omit — every field is therefore optional here. */
export interface LiveSessionItem {
  hidden?: boolean
  id?: string
  last_active?: number
  message_count?: number
  model?: string
  preview?: string
  profile?: string
  session_key?: string
  source?: string
  started_at?: number
  title?: string
}

export interface LiveSessionSnapshot {
  sessions?: LiveSessionItem[]
}

export interface LiveSessionsReconcileOptions {
  /** Opaque registry connection the snapshot came from. Rows MUST carry it:
   *  session-scoped RPC owner resolution reads `connection_id` off the row,
   *  and a missing/wrong stamp routes the RPC to the wrong backend (#102792
   *  class). */
  connectionId: string
  /** Profile key the poll was keyed on; the fallback owner stamp when the
   *  snapshot doesn't name a profile per item. */
  profileKey: string
  /** Injectable clock for the timestamp fallbacks (tests). */
  nowMs?: number
}

// Sources the stored-list sidebar never renders (cron/kanban/subagent/tool +
// every messaging platform, which each own their section). A live row would
// land in Recents, so a live session from one of these must not surface here
// either — the live group obeys the same exclusion as the DB-backed slice.
const EXCLUDED_SOURCE_SET = new Set(SIDEBAR_EXCLUDED_SOURCES)

function trimToNull(value: string | null | undefined): null | string {
  const trimmed = value?.trim()

  return trimmed || null
}

function rowEquals(a: SessionInfo, b: SessionInfo): boolean {
  return (
    a.id === b.id &&
    a.title === b.title &&
    a.preview === b.preview &&
    a.source === b.source &&
    a.model === b.model &&
    a.message_count === b.message_count &&
    a.started_at === b.started_at &&
    a.last_active === b.last_active &&
    a.profile === b.profile &&
    a.is_default_profile === b.is_default_profile &&
    a.connection_id === b.connection_id
  )
}

function snapshotsEqual(a: readonly SessionInfo[], b: readonly SessionInfo[]): boolean {
  return a.length === b.length && a.every((row, index) => rowEquals(row, b[index]))
}

/** True when this live session is already represented by a row the sidebar
 *  knows: a stored slice row matched by STORED id — `sessionMatchesStoredId`
 *  also counts lineage roots and compression tips, so a rotated id never
 *  double-lists — an unlisted-draft owner stub, or a delete/archive tombstone
 *  still awaiting backend confirmation.
 *
 *  Deliberately NOT the owner ladder: `ownerLookupSessionRows()` reads the
 *  stored slices and the draft stubs, never this atom. A live row's owner
 *  reaches resolution through `onResumeSession(session.id, session)` — the
 *  callback carries the stamped row — which is why every live row must keep
 *  its `connection_id` and `profile` stamps.
 *
 *  An open session TILE is deliberately NOT a representation here either: a
 *  row click opens `in-place`, which loads into main (`app/open-session.ts`),
 *  never a tile, and this renderer's own draft tiles are covered by their owner
 *  stubs (or the optimistic row) anyway. Counting tiles would drop the row of
 *  the session the user just clicked while it still has no stored row — the
 *  group would look like it ate the session. A stored row whose tile is open
 *  stays listed for the same reason (`sessionsToKeep`). */
function isAlreadyRepresented(sessionKey: string): boolean {
  const tombstones = $removedSessionIds.get()

  if (tombstones.has(sessionKey)) {
    return true
  }

  const lists: readonly (readonly SessionInfo[])[] = [
    $sessions.get(),
    $cronSessions.get(),
    $messagingSessions.get(),
    $unlistedSessionOwnerRows.get()
  ]

  return lists.some(list => list.some(row => sessionMatchesStoredId(row, sessionKey)))
}

/**
 * What the sidebar renders: `$liveSessions` minus anything the STORED slices
 * represent right now.
 *
 * The reconciler above already applies this at write time, but it only runs on
 * the 1.5s `session.active_list` poll — while the stored list refresh is
 * trailing-throttled (`SESSIONS_LIST_TICK_GAP_MS`, 10s) and deferred further
 * during a typing burst. In that window the first prompt has persisted the row
 * and Recents is already showing it, so filtering only at write time would
 * render the same session twice. Reading through a computed closes the window
 * to a single React pass.
 */
export const $visibleLiveSessions = computed(
  [$liveSessions, $sessions, $cronSessions, $messagingSessions, $unlistedSessionOwnerRows, $removedSessionIds],
  rows => {
    const visible = rows.filter(row => !isAlreadyRepresented(row.id))

    // Keep the array identity when nothing was filtered: React and the
    // per-list memo caches key on it.
    return visible.length === rows.length ? rows : visible
  }
)

/**
 * Turn a `session.active_list` snapshot into the live-group rows.
 *
 * `response.sessions === undefined` means the backend gave NO INFORMATION
 * (older gateways may reject the method outright; the poll's catch skips this
 * call on failure, and a payload without the field must be treated the same
 * way): the atom is left completely untouched — a failed or degraded request
 * must never clear or prune rows that a later GOOD snapshot will re-assert.
 * Only an explicit empty array clears the group (the gateway authoritatively
 * reports nothing live).
 */
export function reconcileLiveSessions(
  response: LiveSessionSnapshot,
  options: LiveSessionsReconcileOptions
): SessionInfo[] {
  if (response.sessions === undefined) {
    return $liveSessions.get()
  }

  const connectionId = options.connectionId.trim()
  const fallbackProfile = normalizeProfileKey(options.profileKey)
  const nowSec = (options.nowMs ?? Date.now()) / 1000

  const next: SessionInfo[] = []
  const seen = new Set<string>()

  for (const item of response.sessions) {
    const sessionKey = trimToNull(item.session_key)

    // No stored id → nothing to resume, nothing to dedupe against. (The
    // server already filters `_finalized` sessions out of the snapshot —
    // tui_gateway methods_session `session.active_list` — so there is no
    // finalized-equivalent id left to drop client-side.)
    if (!sessionKey || seen.has(sessionKey)) {
      continue
    }

    if (item.hidden === true) {
      continue
    }

    // Missing source on an older backend is UNKNOWN, not excluded: #50799 is
    // exactly about sessions the stored list can't see, and dropping unknown
    // sources would silently disable the fix on the backends that need it
    // most. The remaining guards (hidden, dedupe, tombstones) stay protective.
    const source = normalizeSessionSource(item.source)

    if (source && EXCLUDED_SOURCE_SET.has(source)) {
      continue
    }

    if (isAlreadyRepresented(sessionKey)) {
      continue
    }

    seen.add(sessionKey)

    const lastActive = typeof item.last_active === 'number' ? item.last_active : nowSec
    const profile = normalizeProfileKey(item.profile || fallbackProfile)

    next.push({
      connection_id: connectionId,
      cwd: null,
      ended_at: null,
      id: sessionKey,
      input_tokens: 0,
      is_active: true,
      is_default_profile: profile === 'default',
      last_active: lastActive,
      message_count: item.message_count ?? 0,
      model: trimToNull(item.model),
      output_tokens: 0,
      preview: trimToNull(item.preview),
      profile,
      source,
      started_at: typeof item.started_at === 'number' ? item.started_at : lastActive,
      title: trimToNull(item.title),
      tool_call_count: 0
    })
  }

  next.sort((a, b) => b.last_active - a.last_active)

  const prev = $liveSessions.get()

  // Preserve array reference identity when nothing changed: per-list memo
  // caches key on the array reference and the poll re-runs every 1.5s.
  if (!snapshotsEqual(prev, next)) {
    $liveSessions.set(next)
  }

  return $liveSessions.get()
}

/** Connection/profile switch wipe: the next backend re-mints its own live set,
 *  so nothing here survives a gateway swap (call alongside the other
 *  gateway-bound clears in store/gateway-switch.ts). */
export function clearLiveSessions(): void {
  if ($liveSessions.get().length) {
    $liveSessions.set([])
  }
}
