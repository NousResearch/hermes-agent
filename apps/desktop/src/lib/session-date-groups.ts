import { type SidebarSessionEntry } from '@/lib/session-branch-tree'
import { calendarBucket, HOUR, localeWeekStartDay, MINUTE, SECOND, type SessionBucket } from '@/lib/time'

// A flat list row is either a chronological date-bucket divider or a session
// entry. Interleaving these lets the flat list (and the virtualizer) render
// date separators inline without a second layer of nesting.
export type SidebarListRow =
  | { bucket: SessionBucket; key: string; kind: 'divider' }
  | { entry: SidebarSessionEntry; kind: 'session' }

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
  weekStartsOn = localeWeekStartDay()
): SidebarListRow[] {
  const rows: SidebarListRow[] = []
  const emitted = new Set<string>()
  const cutoff = headRunCutoffMs(entries, nowMs, weekStartsOn)
  let lastKey: null | string = null

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
        rows.push({ bucket, key: bucket.key, kind: 'divider' })
      }
    }

    rows.push({ entry, kind: 'session' })
  }

  return rows
}

// Wrap entries as plain session rows (no dividers) so the ungrouped path shares
// the same `SidebarListRow[]` shape as the grouped one.
export function toSessionRows(entries: readonly SidebarSessionEntry[]): SidebarListRow[] {
  return entries.map(entry => ({ entry, kind: 'session' }))
}
