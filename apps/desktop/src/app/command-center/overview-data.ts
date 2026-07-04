// Pure derivations for the command center's cross-profile "All profiles"
// overview. Everything here folds ALREADY-FETCHED store state ($sessions +
// working/attention sets + the kanban board atom) into render-ready rows —
// no fetching, no sockets, so the overview stays REST-only by construction.

import { KANBAN_COLUMNS, type KanbanColumnName } from '@/store/kanban'
import { normalizeProfileKey, sortByProfileOrder } from '@/store/profile'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'
import type { KanbanBoardPayload, KanbanCard } from '@/types/kanban'

export interface ProfileOverviewRow {
  /** Normalized profile key (see {@link normalizeProfileKey}). */
  key: string
  isDefault: boolean
  /** Sessions of this profile currently streaming a turn. */
  workingCount: number
  /** Sessions of this profile blocked on user input. */
  attentionCount: number
  /** Listable conversation total when the aggregator reported one, else the
   *  loaded in-memory count (same fallback the sidebar uses). */
  sessionCount: number
  /** Most recent activity across the profile's loaded sessions (epoch secs). */
  lastActive: null | number
}

export interface ProfileOverviewInput {
  sessions: SessionInfo[]
  profiles: ProfileInfo[]
  /** Rail drag order — rows mirror the rail so the overview reads spatially. */
  profileOrder: string[]
  workingIds: readonly string[]
  attentionIds: readonly string[]
  /** Per-profile totals from /api/profiles/sessions (may be empty). */
  profileTotals: Record<string, number>
}

interface Bucket {
  attention: number
  count: number
  lastActive: number
  working: number
}

/** One row per profile, in rail order (default first, then named profiles in
 *  the user's drag order). Profiles that only appear on session rows (e.g. the
 *  profile list hasn't refreshed yet) still get a row, appended alphabetically,
 *  so activity is never silently dropped. */
export function buildProfileOverviewRows(input: ProfileOverviewInput): ProfileOverviewRow[] {
  const workingSet = new Set(input.workingIds)
  const attentionSet = new Set(input.attentionIds)

  const buckets = new Map<string, Bucket>()

  const ensure = (key: string): Bucket => {
    let bucket = buckets.get(key)

    if (!bucket) {
      bucket = { attention: 0, count: 0, lastActive: 0, working: 0 }
      buckets.set(key, bucket)
    }

    return bucket
  }

  for (const session of input.sessions) {
    const bucket = ensure(normalizeProfileKey(session.profile))

    bucket.count += 1

    // Same id semantics as the sidebar row indicators: working/attention sets
    // hold live stored ids, which is what the aggregated list surfaces. A
    // session can be in BOTH sets (turn running AND blocked on input) — mirror
    // deriveProfileActivity's precedence so the rail badge and this overview
    // never disagree: needs-input wins, the session counts in one bucket only.
    if (attentionSet.has(session.id)) {
      bucket.attention += 1
    } else if (workingSet.has(session.id)) {
      bucket.working += 1
    }

    const timestamp = session.last_active || session.started_at || 0

    if (timestamp > bucket.lastActive) {
      bucket.lastActive = timestamp
    }
  }

  const defaultProfile = input.profiles.find(profile => profile.is_default)
  const defaultKey = defaultProfile ? normalizeProfileKey(defaultProfile.name) : 'default'

  const namedKeys = sortByProfileOrder(
    input.profiles.filter(profile => !profile.is_default),
    input.profileOrder
  ).map(profile => normalizeProfileKey(profile.name))

  const orderedKeys = [defaultKey, ...namedKeys.filter(key => key !== defaultKey)]
  const known = new Set(orderedKeys)

  // Session-only profiles (aggregator knows them, the profile list doesn't yet).
  const extras = [...buckets.keys()].filter(key => !known.has(key)).sort((a, b) => a.localeCompare(b))

  return [...orderedKeys, ...extras].map(key => {
    const bucket = buckets.get(key)

    return {
      key,
      isDefault: key === defaultKey,
      workingCount: bucket?.working ?? 0,
      attentionCount: bucket?.attention ?? 0,
      sessionCount: input.profileTotals[key] ?? bucket?.count ?? 0,
      lastActive: bucket?.lastActive ? bucket.lastActive : null
    }
  })
}

export interface KanbanBoardSummary {
  /** Per-status counts in canonical column order (zero-filled). */
  counts: Array<{ count: number; name: KanbanColumnName }>
  total: number
  /** Cards waiting on a human — review first, then blocked, highest priority
   *  first, oldest first within a priority. Capped for chip rendering. */
  hotCards: KanbanCard[]
}

const HOT_STATUS_RANK: Record<string, number> = { review: 0, blocked: 1 }

export function summarizeKanbanBoard(board: KanbanBoardPayload, maxHotCards = 5): KanbanBoardSummary {
  const tasksByColumn = new Map(board.columns.map(column => [column.name, column.tasks]))

  const counts = KANBAN_COLUMNS.map(name => ({
    count: tasksByColumn.get(name)?.length ?? 0,
    name
  }))

  const total = board.columns.reduce((acc, column) => acc + column.tasks.length, 0)

  const hotCards = [...(tasksByColumn.get('review') ?? []), ...(tasksByColumn.get('blocked') ?? [])].sort(
    (a, b) =>
      (HOT_STATUS_RANK[a.status] ?? 2) - (HOT_STATUS_RANK[b.status] ?? 2) ||
      b.priority - a.priority ||
      a.created_at - b.created_at
  )

  return { counts, total, hotCards: hotCards.slice(0, maxHotCards) }
}
