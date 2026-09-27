/**
 * Card presentation — what the face (and the drawer's header and description)
 * reads for a task.
 *
 * On the FLEET board, the sync adapter (conductors' scripts/
 * fleet_kanban_remote.py, `refresh_local_presentation`) projects its own
 * bookkeeping INTO a task's local title and body, so every Hermes surface sees
 * it without a schema change:
 *
 *   title: `[Sync pending] Real title`               (also conflict / error)
 *   body:  `<!-- fleet-kanban:meta -->` ⏎
 *          `> Fleet: revision 6 | point Conductor: … | canonical status: ready` ⏎
 *          `<!-- /fleet-kanban:meta -->` ⏎⏎ `Real body`
 *
 * That is internal synchronization state, not what an operator scans a board
 * for, so THERE the face reads through the decoration and the drawer keeps the
 * lifted lines. The adapter only ever installs on the board whose slug is
 * `fleet` (`fleet_kanban_sqlite.FLEET_BOARD`), and that slug is the only
 * verified context: every other board is shown literally — a title that
 * happens to start with "[Sync pending]" there is just a title, and its
 * `tenant` is a tenant, not a fleet node. Recognition on the fleet board is
 * exact-match on the adapter's own markers, mirroring its strip rules: a title
 * that merely mentions "[Sync pending]" mid-sentence, or a body whose marker is
 * never closed, is left byte-for-byte alone.
 *
 * Board-independent: `latest_summary` is only the newest run summary, and a
 * manual status change (drawer, dashboard) writes the backend's administrative
 * note into that slot. It says nothing about the work, so it never stands in
 * for a summary.
 */

import type { KanbanTask } from './types'

/** Slug of the one board the fleet sync adapter mirrors. */
export const FLEET_BOARD = 'fleet'

/** Whether `slug` names the fleet board — the verified context for reading
 *  through the adapter's decoration. */
export const isFleetBoard = (slug: null | string | undefined): boolean => slug === FLEET_BOARD

export type SyncState = 'conflict' | 'error' | 'pending'

const SYNC_PREFIXES: ReadonlyArray<readonly [SyncState, string]> = [
  ['pending', '[Sync pending] '],
  ['conflict', '[Sync conflict] '],
  ['error', '[Sync error] ']
]

const META_START = '<!-- fleet-kanban:meta -->'
const META_END = '<!-- /fleet-kanban:meta -->'

export interface CardFace {
  /** Body with the meta block lifted off; null when nothing readable remains. */
  body: null | string
  /** The meta block's content lines, for the drawer's detail surface. */
  meta: string[]
  /** What the face shows under the title: the latest real work summary, else the body. */
  summary: null | string
  /** Outbox state the title prefix encoded, when decorated. */
  syncState: null | SyncState
  /** Title with the sync prefix lifted off. */
  title: string
}

/** The backend's note for a manual status change (`plugin_api.py`,
 *  `status changed to <status> (dashboard/direct)`). */
export const isAdminSummary = (summary: string): boolean =>
  /^status changed to \w+ \(dashboard\/direct\)$/.test(summary)

/** `latest_summary` when it is a real work summary; null for the
 *  administrative note or nothing. */
export function workSummary(summary: null | string | undefined): null | string {
  return summary && !isAdminSummary(summary) ? summary : null
}

/** Split a sync prefix off the title — only at the very start, only exact. */
export function splitSyncTitle(title: string): { syncState: null | SyncState; title: string } {
  for (const [state, prefix] of SYNC_PREFIXES) {
    if (title.startsWith(prefix)) {
      return { syncState: state, title: title.slice(prefix.length) }
    }
  }

  return { syncState: null, title }
}

/** Split the leading meta block off the body. The adapter composes
 *  `block + "\n\n" + body` (bare block when the body is empty), and strips the
 *  same `\n\n` / `\n` seam back off — so do we. */
export function splitMetaBlock(body: null | string | undefined): { body: null | string; meta: string[] } {
  if (!body) {
    return { body: null, meta: [] }
  }

  if (!body.startsWith(META_START)) {
    return { body, meta: [] }
  }

  const end = body.indexOf(META_END)

  if (end === -1) {
    return { body, meta: [] }
  }

  const meta = body
    .slice(META_START.length, end)
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)

  let rest = body.slice(end + META_END.length)

  if (rest.startsWith('\n\n')) {
    rest = rest.slice(2)
  } else if (rest.startsWith('\n')) {
    rest = rest.slice(1)
  }

  return { body: rest || null, meta }
}

/** What the card face (and the drawer's header/description) should read.
 *  `fleet` is the verified context (see `isFleetBoard`); off the fleet board
 *  the title and body are literal. */
export function cardFace(task: Pick<KanbanTask, 'body' | 'latest_summary' | 'title'>, fleet: boolean): CardFace {
  const { syncState, title } = fleet ? splitSyncTitle(task.title) : { syncState: null, title: task.title }
  const { body, meta } = fleet ? splitMetaBlock(task.body) : { body: task.body || null, meta: [] }

  return { body, meta, summary: workSummary(task.latest_summary) ?? body, syncState, title }
}
