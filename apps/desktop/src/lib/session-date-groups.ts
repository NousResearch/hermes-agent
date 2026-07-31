import { type SidebarSessionEntry } from '@/lib/session-branch-tree'
import { calendarBucket, localeWeekStartDay, SECOND, type SessionBucket } from '@/lib/time'

// A flat list row is either a divider or a session entry. Interleaving these
// lets the flat list (and the virtualizer) render separators inline without a
// second layer of nesting. A divider either names a calendar bucket (resolved
// against the locale's labels at render time) or carries its own label.
export type SidebarListRow =
  | { bucket: SessionBucket; key: string; kind: 'divider'; rowKey?: string }
  | { entry: SidebarSessionEntry; kind: 'session' }
  | { key: string; kind: 'divider'; label: string; rowKey?: string }

// The row's own age label reads from `last_active || started_at`; bucket off the
// same value so a divider lines up with what the row actually shows.
const recencyMs = (entry: SidebarSessionEntry): number =>
  (entry.session.last_active || entry.session.started_at || 0) * SECOND

// Insert a divider before every populated calendar group, including the first.
// Preserve the caller's exact order. If a bucket reappears after another one,
// its segment gets a unique React row key while sharing the same collapse key.
// Branch children inherit their parent segment and never trigger a divider.
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

    rows.push({ entry, kind: 'session' })
  }

  return rows
}

// Split into two runs — still busy, and everything else — under the same
// dividers the date grouping uses. Branch children ride with their parent, as
// they do there. Entries arrive already ordered, so each run is contiguous.
export function groupEntriesByStatus(
  entries: readonly SidebarSessionEntry[],
  isWorking: (entry: SidebarSessionEntry) => boolean,
  labels: { done: string; working: string }
): SidebarListRow[] {
  const working: SidebarSessionEntry[] = []
  const done: SidebarSessionEntry[] = []
  let cluster = done

  for (const entry of entries) {
    if (!entry.branchStem) {
      cluster = isWorking(entry) ? working : done
    }

    cluster.push(entry)
  }

  return [
    ...(working.length ? [{ key: 'status:working', kind: 'divider' as const, label: labels.working }] : []),
    ...toSessionRows(working),
    ...(done.length ? [{ key: 'status:done', kind: 'divider' as const, label: labels.done }] : []),
    ...toSessionRows(done)
  ]
}

// Wrap entries as plain session rows (no dividers) so the ungrouped path shares
// the same `SidebarListRow[]` shape as the grouped one.
export function toSessionRows(entries: readonly SidebarSessionEntry[]): SidebarListRow[] {
  return entries.map(entry => ({ entry, kind: 'session' }))
}

// Keep every divider mounted so a collapsed group always has a disclosure the
// user can reopen. Session rows after a collapsed divider stay hidden until the
// next divider.
export function filterCollapsedDateGroupRows(
  rows: readonly SidebarListRow[],
  collapsedKeys: ReadonlySet<string>
): SidebarListRow[] {
  const visible: SidebarListRow[] = []
  let collapsed = false

  for (const row of rows) {
    if (row.kind === 'divider') {
      collapsed = collapsedKeys.has(row.key)
      visible.push(row)

      continue
    }

    if (!collapsed) {
      visible.push(row)
    }
  }

  return visible
}
