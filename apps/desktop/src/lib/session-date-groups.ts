import { type SidebarSessionEntry } from '@/lib/session-branch-tree'
import { calendarBucket, localeWeekStartDay, SECOND, type SessionBucket } from '@/lib/time'

// Dividers and conversations share one flat row model so the virtualizer can
// measure exactly what is visible. Grouped session rows carry their technical
// key; ungrouped rows intentionally do not.
export type SidebarListRow =
  | { bucket: SessionBucket; key: string; kind: 'divider'; rowKey: string }
  | { entry: SidebarSessionEntry; groupKey?: string; kind: 'session' }

// Keep grouping aligned with the timestamp already used by the row age label.
// Renaming, pinning and other presentation-only mutations never enter here.
const recencyMs = (entry: SidebarSessionEntry): number =>
  (entry.session.last_active || entry.session.started_at || 0) * SECOND

// Preserve the caller's exact order. A bucket that reappears after another
// bucket starts a new visual segment with a row identity anchored to that
// segment's first session, while `key` remains the shared persistence key.
// Branch children inherit their parent's segment even when their own stored
// timestamp is older.
export function groupEntriesByRecency(
  entries: readonly SidebarSessionEntry[],
  nowMs = Date.now(),
  weekStartsOn = localeWeekStartDay()
): SidebarListRow[] {
  const rows: SidebarListRow[] = []
  let currentBucket: SessionBucket | undefined

  for (const entry of entries) {
    if (!entry.branchStem || !currentBucket) {
      const bucket = calendarBucket(recencyMs(entry) / SECOND, nowMs, weekStartsOn)

      if (!currentBucket || currentBucket.key !== bucket.key) {
        rows.push({
          bucket,
          key: bucket.key,
          kind: 'divider',
          rowKey: `date-divider:${bucket.key}:${entry.session.id}`
        })
      }

      currentBucket = bucket
    }

    rows.push({ entry, groupKey: currentBucket.key, kind: 'session' })
  }

  return rows
}

export function sessionDateGroupKeys(rows: readonly SidebarListRow[]): string[] {
  return [...new Set(rows.flatMap(row => (row.kind === 'divider' ? [row.key] : [])))]
}

// Filter before virtualisation so hidden conversations contribute neither rows
// nor measured height. Dividers remain mounted and keyboard reachable.
export function visibleSessionDateGroupRows(
  rows: readonly SidebarListRow[],
  collapsedGroupKeys: ReadonlySet<string>
): SidebarListRow[] {
  if (collapsedGroupKeys.size === 0) {
    return [...rows]
  }

  return rows.filter(row => row.kind === 'divider' || !row.groupKey || !collapsedGroupKeys.has(row.groupKey))
}

export function toSessionRows(entries: readonly SidebarSessionEntry[]): SidebarListRow[] {
  return entries.map(entry => ({ entry, kind: 'session' }))
}
