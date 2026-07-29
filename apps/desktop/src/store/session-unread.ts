import { Codecs, persistentAtom } from '@/lib/persisted'
import { stableArray } from '@/lib/stable-array'
import type { SessionInfo } from '@/types/hermes'

import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  sessionMatchesStoredId,
  sessionPinId
} from './session'

/**
 * PERSISTED UNREAD ("finished — unread" green dot) — the durable layer under
 * the transient `$unreadFinishedSessionIds` atom, ported from the webui's
 * proven design (sessions.js: viewed-count watermarks + completion markers).
 *
 * Two persisted records, both keyed by the DURABLE lineage id (sessionPinId)
 * so state survives auto-compression's session-id rotation — same rule as
 * `$sessionColorOverrides`:
 *
 *  1. SEEN WATERMARKS — the message_count last acknowledged for a session
 *     (set when the user opens it, kept current while it stays selected, and
 *     seeded on FIRST SIGHT of an unknown session so a fresh install doesn't
 *     paint every row green). A row whose live `message_count` exceeds its
 *     watermark is unread — this is what reconstructs the green dot for a
 *     session that finished while the app was CLOSED, which the live
 *     busy→idle edge in session-states.ts can never replay after a restart.
 *
 *  2. EXPLICIT MARKERS — the live busy→idle edge, persisted. Covers the gap
 *     where a turn finishes in the background but the sidebar list hasn't
 *     refreshed its message_count yet (and any completion that adds no
 *     stored messages), so a restart right after a finish keeps the dot.
 *
 * Keys are durable ids WITHOUT profile scoping on purpose: session ids are
 * unique across profiles, and unread only ever PAINTS for rows present in the
 * loaded lists (which are profile-scoped), so another profile's markers stay
 * dormant instead of leaking — and survive a profile round-trip (the webui
 * needed explicit per-profile scoping, commit 1a64fc68, because its markers
 * painted by id lookup alone).
 *
 * Scope of watermark-based unread: chat + cron rows. Messaging rows get their
 * message_count bumped by inbound human messages, so count-over-watermark
 * would paint "finished — unread" on every new inbound text; they keep only
 * the explicit live-edge markers (now persisted across restarts too).
 */

export const $sessionSeenCounts = persistentAtom<Record<string, number>>(
  'hermes.desktop.sessionSeenCounts',
  {},
  Codecs.json(sanitizeSeenCounts)
)

export const $unreadFinishedMarkers = persistentAtom<string[]>(
  'hermes.desktop.unreadFinishedSessions',
  [],
  Codecs.stringArray
)

function sanitizeSeenCounts(value: unknown): Record<string, number> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1])
    )
  )
}

// Watermarks accrete one entry per session ever seen; keep the record bounded.
// Past the cap, entries for sessions no longer in any loaded list (and not
// explicitly marked) are dropped — a pruned session reseeds on next sight,
// which reads as "not unread", the safe default.
const SEEN_COUNTS_CAP = 2000

const rowsFor = (lists: readonly SessionInfo[][]): SessionInfo[] => lists.flat()

const isSelected = (row: SessionInfo, selected: null | string): boolean =>
  Boolean(selected && sessionMatchesStoredId(row, selected))

/** Durable id for a stored id, when the row is loaded; the stored id itself
 *  otherwise (sessionPinId falls back to the live id anyway). */
function durableIdForStoredId(storedSessionId: string): string {
  const row = rowsFor([$sessions.get(), $cronSessions.get(), $messagingSessions.get()]).find(s =>
    sessionMatchesStoredId(s, storedSessionId)
  )

  return row ? sessionPinId(row) : storedSessionId
}

/** LIVE-EDGE WRITER — called by session-states.ts on a background busy→idle
 *  transition. Flags the transient atom (immediate paint) AND persists the
 *  marker so the dot survives a restart. */
export function markSessionUnreadFinished(storedSessionId: string): void {
  const current = $unreadFinishedSessionIds.get()

  if (!current.includes(storedSessionId)) {
    $unreadFinishedSessionIds.set([...current, storedSessionId])
  }

  const durableId = durableIdForStoredId(storedSessionId)
  const markers = $unreadFinishedMarkers.get()

  if (!markers.includes(durableId)) {
    $unreadFinishedMarkers.set([...markers, durableId])
  }
}

/** ACK — the user opened (or is looking at) this session: watermark := its
 *  current message_count, and any explicit marker is retired. */
function ackSessionRow(row: SessionInfo): void {
  const durableId = sessionPinId(row)

  if (Number.isFinite(row.message_count)) {
    const seen = $sessionSeenCounts.get()

    if (seen[durableId] !== row.message_count) {
      $sessionSeenCounts.set({ ...seen, [durableId]: row.message_count })
    }
  }

  const markers = $unreadFinishedMarkers.get()
  const next = markers.filter(id => id !== durableId && !sessionMatchesStoredId(row, id))

  if (next.length !== markers.length) {
    $unreadFinishedMarkers.set(next)
  }
}

/** Clear persisted unread for a stored id even when its row isn't loaded —
 *  the marker alone can be retired; the watermark needs the row's count. */
export function ackStoredSessionId(storedSessionId: null | string): void {
  if (!storedSessionId) {
    return
  }

  const row = rowsFor([$sessions.get(), $cronSessions.get(), $messagingSessions.get()]).find(s =>
    sessionMatchesStoredId(s, storedSessionId)
  )

  if (row) {
    ackSessionRow(row)

    return
  }

  const markers = $unreadFinishedMarkers.get()
  const next = markers.filter(id => id !== storedSessionId)

  if (next.length !== markers.length) {
    $unreadFinishedMarkers.set(next)
  }
}

/** Seed/refresh watermarks from freshly loaded lists: the SELECTED session
 *  tracks its live count (it's on screen — nothing there is unread), and a
 *  session never seen before seeds at its current count instead of lighting
 *  up green on first sight. Known, unselected rows are left alone — the gap
 *  between their watermark and live count IS the unread signal. */
function ingestRows(rows: readonly SessionInfo[]): void {
  const selected = $selectedStoredSessionId.get()
  const seen = $sessionSeenCounts.get()
  let next: null | Record<string, number> = null

  for (const row of rows) {
    if (!Number.isFinite(row.message_count)) {
      continue
    }

    const durableId = sessionPinId(row)

    if (isSelected(row, selected)) {
      if (seen[durableId] !== row.message_count) {
        ;(next ??= { ...seen })[durableId] = row.message_count
      }

      // Any lingering explicit marker for the on-screen session is stale.
      const markers = $unreadFinishedMarkers.get()
      const kept = markers.filter(id => id !== durableId && !sessionMatchesStoredId(row, id))

      if (kept.length !== markers.length) {
        $unreadFinishedMarkers.set(kept)
      }
    } else if (!(durableId in seen) && !(next && durableId in next)) {
      ;(next ??= { ...seen })[durableId] = row.message_count
    }
  }

  if (next) {
    $sessionSeenCounts.set(pruneSeenCounts(next, rows))
  }
}

function pruneSeenCounts(seen: Record<string, number>, rows: readonly SessionInfo[]): Record<string, number> {
  const keys = Object.keys(seen)

  if (keys.length <= SEEN_COUNTS_CAP) {
    return seen
  }

  const keep = new Set<string>($unreadFinishedMarkers.get())

  for (const row of rows) {
    keep.add(sessionPinId(row))
  }

  return Object.fromEntries(keys.filter(key => keep.has(key)).map(key => [key, seen[key]]))
}

/** Recompute the transient unread atom from persisted state + loaded rows.
 *  Runs after every list refresh — including the first one after a cold
 *  start, which is what brings green dots back from a restart. Stored ids
 *  already flagged by the live edge but not (yet) present in any list — a
 *  brand-new session's first turn isn't flushed to the list until persisted —
 *  are preserved, not dropped. */
function recomputeUnread(): void {
  const selected = $selectedStoredSessionId.get()
  const markers = $unreadFinishedMarkers.get()
  const seen = $sessionSeenCounts.get()
  const unread: string[] = []

  // Watermark + marker unread for chat and cron rows.
  for (const row of rowsFor([$sessions.get(), $cronSessions.get()])) {
    if (isSelected(row, selected)) {
      continue
    }

    const durableId = sessionPinId(row)
    const watermark = seen[durableId]

    const exceedsWatermark =
      typeof watermark === 'number' && Number.isFinite(row.message_count) && row.message_count > watermark

    if (exceedsWatermark || markers.includes(durableId) || markers.includes(row.id)) {
      unread.push(row.id)
    }
  }

  // Messaging rows: explicit (live-edge) markers only — see header comment.
  for (const row of $messagingSessions.get()) {
    if (!isSelected(row, selected) && (markers.includes(sessionPinId(row)) || markers.includes(row.id))) {
      unread.push(row.id)
    }
  }

  // Preserve live-edge ids whose row isn't loaded yet.
  const loadedRows = rowsFor([$sessions.get(), $cronSessions.get(), $messagingSessions.get()])

  for (const id of $unreadFinishedSessionIds.get()) {
    if (id !== selected && !unread.includes(id) && !loadedRows.some(row => sessionMatchesStoredId(row, id))) {
      unread.push(id)
    }
  }

  const current = $unreadFinishedSessionIds.get()
  const next = stableArray(current, unread)

  if (next !== current) {
    $unreadFinishedSessionIds.set([...next])
  }
}

function onListChange(): void {
  ingestRows(rowsFor([$sessions.get(), $cronSessions.get(), $messagingSessions.get()]))
  recomputeUnread()
}

// Wiring: module-scope listeners, same pattern as session-states.ts's
// $selectedStoredSessionId listener. Loaded via session-states.ts (the live
// edge imports markSessionUnreadFinished), so it's active wherever unread is.
$sessions.listen(onListChange)
$cronSessions.listen(onListChange)
$messagingSessions.listen(onListChange)

// Opening a session acks it durably (the transient atom is already cleared
// synchronously by setSelectedStoredSessionId — this is the persisted half).
$selectedStoredSessionId.listen(ackStoredSessionId)
