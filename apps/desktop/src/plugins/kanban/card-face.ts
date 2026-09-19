/**
 * Fleet card presentation. The fleet sync adapter (conductors' scripts/
 * fleet_kanban_remote.py, `refresh_local_presentation`) projects its own
 * bookkeeping INTO a local task's title and body, so every Hermes surface sees
 * it without a schema change:
 *
 *   title: `[Sync pending] Real title`               (also conflict / error)
 *   body:  `<!-- fleet-kanban:meta -->` ⏎
 *          `> Fleet: revision 6 | point Conductor: … | canonical status: ready` ⏎
 *          `<!-- /fleet-kanban:meta -->` ⏎⏎ `Real body`
 *
 * That is internal synchronization state, not what an operator scans a board
 * for. The card face reads THROUGH the decoration (readable title, readable
 * body); the drawer keeps the lifted lines as its detail surface. Recognition
 * is exact-match on the adapter's own markers, mirroring its strip rules: a
 * title that merely mentions "[Sync pending]" mid-sentence, or a body whose
 * marker is never closed, is left byte-for-byte alone.
 */

import type { KanbanTask } from './types'

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
  /** Outbox state the title prefix encoded, when decorated. */
  syncState: null | SyncState
  /** Title with the sync prefix lifted off. */
  title: string
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

/** What the card face (and the drawer's header/description) should read. */
export function cardFace(task: Pick<KanbanTask, 'body' | 'title'>): CardFace {
  const { syncState, title } = splitSyncTitle(task.title)
  const { body, meta } = splitMetaBlock(task.body)

  return { body, meta, syncState, title }
}
