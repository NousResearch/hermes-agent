import { type SidebarSessionEntry } from '@/lib/session-branch-tree'
import {
  calendarBucket,
  DAY,
  HOUR,
  localeWeekStartDay,
  MINUTE,
  nominalDayStart,
  SECOND,
  type SessionBucket
} from '@/lib/time'
import {
  type SessionAttentionGroup,
  sessionAttentionGroup,
  type SessionDotState,
  sessionStatusRank
} from '@/store/session-dot-state'

// A flat list row is either a divider or a session entry. Interleaving these
// lets the flat list (and the virtualizer) render separators inline without a
// second layer of nesting. A divider either names a calendar bucket (resolved
// against the locale's labels at render time) or carries its own label.
export type SidebarListRow =
  | { bucket: SessionBucket; key: string; kind: 'divider' }
  | { entry: SidebarSessionEntry; kind: 'session' }
  | { count?: number; key: string; kind: 'divider'; label: string }

// The row's own age label reads from `last_active || started_at`; bucket off the
// same value so a divider lines up with what the row actually shows.
const recencyMs = (entry: SidebarSessionEntry): number =>
  (entry.session.last_active || entry.session.started_at || 0) * SECOND

// Aim the head at "the most recent handful". A break shorter than
// MIN_RUN_BREAK_MS never counts as one — that would slice a rapid-fire burst —
// and a silence longer than MAX_RUN_GAP_MS always ends the run: without that
// bound a sparse list (a project lane) would chain weeks of stale sessions
// into one giant "recent" head.
const TARGET_HEAD_SESSIONS = 5
const MIN_RUN_BREAK_MS = 30 * MINUTE
const MAX_RUN_GAP_MS = 8 * HOUR

// The unlabelled head is the newest run of sessions, cut at a *real* break in
// activity. Candidate cut points are every gap of at least MIN_RUN_BREAK_MS
// inside the contiguous run (gaps ≤ MAX_RUN_GAP_MS), plus the run's own end;
// among them we pick the one whose head size lands closest (log-scale) to
// TARGET_HEAD_SESSIONS. So the first divider shows up after roughly the most
// recent five sessions — but only ever at a genuine pause, never mid-burst: a
// truly unbroken run stays whole, and an isolated newest session stands alone.
// Runs chain naturally across midnight.
//
// Fuzzy-merge rule: when the cut falls at the run's end and the sessions just
// below it share the head's calendar bucket, the head adds nothing — it's just
// the top of that group. Dissolve it (the first-group rule keeps the top
// unlabelled anyway) so a divider never strands near-identical neighbours,
// e.g. a lone 6-day-old session above a "Last week" label.
//
// Returns the oldest timestamp (ms) still inside the head; -Infinity means the
// whole list is one run, +Infinity means no head (calendar groups own it all).
function headRunCutoffMs(entries: readonly SidebarSessionEntry[], nowMs: number, weekStartsOn: number): number {
  const times = entries
    .filter(entry => !entry.branchStem)
    .map(recencyMs)
    .sort((a, b) => b - a)

  let bestIdx = -1
  let bestScore = Number.POSITIVE_INFINITY
  let runEnded = false

  for (let i = 1; i < times.length; i++) {
    const gap = times[i - 1] - times[i]
    const endsRun = gap > MAX_RUN_GAP_MS

    if (gap >= MIN_RUN_BREAK_MS || endsRun) {
      // `i` sessions would sit above a cut at this gap.
      const score = Math.abs(Math.log(i / TARGET_HEAD_SESSIONS))

      if (score < bestScore) {
        bestScore = score
        bestIdx = i
        runEnded = endsRun
      }
    }

    if (endsRun) {
      break
    }
  }

  if (bestIdx === -1) {
    return Number.NEGATIVE_INFINITY
  }

  if (runEnded) {
    const headBucket = calendarBucket(times[0] / SECOND, nowMs, weekStartsOn)
    const belowBucket = calendarBucket(times[bestIdx] / SECOND, nowMs, weekStartsOn)

    if (headBucket.key === belowBucket.key) {
      return Number.POSITIVE_INFINITY
    }
  }

  return times[bestIdx - 1]
}

// Insert a date divider before each labelled group. The unlabelled head is the
// newest run of sessions (see headRunCutoffMs); below it, groups are coarse
// calendar ranges — earlier today → yesterday → earlier this week → last week
// → earlier this month → month → month + year — one divider per range, never
// one per day. Whatever group happens to render first is also never labelled.
// Branch children inherit their parent cluster's group and never trigger a
// divider, so a parent→branches block never splits.
export function groupEntriesByRecency(
  entries: readonly SidebarSessionEntry[],
  nowMs = Date.now(),
  weekStartsOn = localeWeekStartDay(),
  maxGroups = Number.POSITIVE_INFINITY
): SidebarListRow[] {
  const rows: SidebarListRow[] = []
  const emitted = new Set<string>()
  const cutoff = headRunCutoffMs(entries, nowMs, weekStartsOn)
  let lastKey: null | string = null
  let groups = 1

  for (const entry of entries) {
    // Nested branch rows travel with their parent cluster; they never open a new
    // bucket or move the divider cursor.
    if (entry.branchStem) {
      rows.push({ entry, kind: 'session' })

      continue
    }

    const ms = recencyMs(entry)

    // Head-run sessions are never labelled.
    if (ms >= cutoff) {
      rows.push({ entry, kind: 'session' })
      lastKey = '__recent__'

      continue
    }

    const bucket = calendarBucket(ms / SECOND, nowMs, weekStartsOn)

    if (bucket.key !== lastKey) {
      lastKey = bucket.key
      const alreadyEmitted = emitted.has(bucket.key)

      // Mark it emitted even when skipped so a non-monotonic order (possible
      // inside a project lane) can't later re-label it or collide React keys.
      emitted.add(bucket.key)

      // A divider only ever separates two groups — never label the very first
      // rendered row, whatever group it belongs to.
      if (rows.length > 0 && !alreadyEmitted) {
        if (groups >= maxGroups) {
          break
        }

        groups++
        rows.push({ bucket, key: bucket.key, kind: 'divider' })
      }
    }

    rows.push({ entry, kind: 'session' })
  }

  return rows
}

// Idle is split by how long nobody has touched the session, so last week's
// leftovers stop competing with today's work. Only Idle ages: a session that
// needs the user or is still running keeps its lane however old it is — age is
// never evidence that work is finished. Days are nominal (4 AM rollover, same
// as the date grouping); each lane holds ages >= its floor, oldest floor first.
export type SessionIdleLane = 'idle' | 'idleDays' | 'idleRecent' | 'idleWeek'

export type SessionStatusLane = Exclude<SessionAttentionGroup, 'idle'> | SessionIdleLane

const IDLE_LANE_MIN_DAYS: readonly (readonly [SessionIdleLane, number])[] = [
  ['idleWeek', 7],
  ['idleDays', 3],
  ['idleRecent', 1],
  ['idle', 0]
]

export const SESSION_STATUS_LANE_ORDER: readonly SessionStatusLane[] = [
  'attention',
  'working',
  'idle',
  'idleRecent',
  'idleDays',
  'idleWeek'
]

// Lanes that start folded until the user opens them: the 7+ day tail is a
// review pile, not something to scan every time the sidebar opens.
const DEFAULT_CLOSED_LIST_GROUPS: ReadonlySet<string> = new Set(['status:idleWeek'])

export const listGroupDefaultOpen = (key: string): boolean => !DEFAULT_CLOSED_LIST_GROUPS.has(key)

/** Idle lane for a session last touched at `lastActiveMs`. A future timestamp
 *  (clock skew) counts as today; a missing one (0) as the oldest lane. */
export function idleLane(lastActiveMs: number, nowMs: number): SessionIdleLane {
  const days = Math.round((nominalDayStart(nowMs) - nominalDayStart(lastActiveMs)) / DAY)

  return IDLE_LANE_MIN_DAYS.find(([, min]) => days >= min)?.[0] ?? 'idle'
}

interface StatusCluster {
  entries: SidebarSessionEntry[]
  rank: number
  state: SessionDotState
}

// A cluster is as fresh as its newest member, so a branch touched today keeps
// an old parent out of the stale lanes.
function clusterLane(cluster: StatusCluster, nowMs: number): SessionStatusLane {
  const group = sessionAttentionGroup(cluster.state)

  return group === 'idle' ? idleLane(Math.max(...cluster.entries.map(recencyMs)), nowMs) : group
}

// Lanes by whose move it is — Needs attention, Working, then Idle by age (today,
// 1–2 days, 3–6 days, 7+ days) — under the same dividers the date grouping
// uses, each with its session count. Inside a lane the louder state leads
// (blocked prompt before unread, draft before idle); otherwise the incoming
// order (the active sort key) is kept. Branch children ride with their parent,
// as they do there, so a cluster moves as one unit.
export function groupEntriesByStatus(
  entries: readonly SidebarSessionEntry[],
  stateOf: (entry: SidebarSessionEntry) => SessionDotState,
  labels: Record<SessionStatusLane, string>,
  nowMs = Date.now()
): SidebarListRow[] {
  const clusters: StatusCluster[] = []

  for (const entry of entries) {
    const last = clusters.at(-1)

    if (entry.branchStem && last) {
      last.entries.push(entry)
    } else {
      const state = stateOf(entry)

      clusters.push({ entries: [entry], rank: sessionStatusRank(state), state })
    }
  }

  const lanes = new Map<SessionStatusLane, StatusCluster[]>(SESSION_STATUS_LANE_ORDER.map(lane => [lane, []]))

  for (const cluster of clusters) {
    lanes.get(clusterLane(cluster, nowMs))!.push(cluster)
  }

  return SESSION_STATUS_LANE_ORDER.flatMap(lane => {
    // Array.prototype.sort is stable, so equal ranks keep the sort key's order.
    const members = lanes.get(lane)!.sort((a, b) => a.rank - b.rank)

    return members.length
      ? [
          { count: members.length, key: `status:${lane}`, kind: 'divider' as const, label: labels[lane] },
          ...toSessionRows(members.flatMap(c => c.entries))
        ]
      : []
  })
}

// Wrap entries as plain session rows (no dividers) so the ungrouped path shares
// the same `SidebarListRow[]` shape as the grouped one.
export function toSessionRows(entries: readonly SidebarSessionEntry[]): SidebarListRow[] {
  return entries.map(entry => ({ entry, kind: 'session' }))
}

/** Drop session rows that sit under a closed divider. The divider itself stays
 *  so the group can be opened again. Sessions above the first divider (the
 *  unlabelled head) are never gated. Returns the input array when nothing is
 *  hidden so callers keep a stable reference. */
export function hideCollapsedGroupRows(
  rows: readonly SidebarListRow[],
  isOpen: (key: string) => boolean
): SidebarListRow[] {
  const out: SidebarListRow[] = []
  let hiding = false

  for (const row of rows) {
    if (row.kind === 'divider') {
      hiding = !isOpen(row.key)
      out.push(row)

      continue
    }

    if (!hiding) {
      out.push(row)
    }
  }

  return out.length === rows.length ? (rows as SidebarListRow[]) : out
}
