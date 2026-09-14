import type { SessionInfo } from '@/hermes'
import type { SidebarRecencyFilter } from '@/store/layout'

import { sessionRecency } from './projects/workspace-groups'

/** Window length per option, in the seconds `sessionRecency` reports. Keyed by
 *  the union so a new option can't compile without its own window. */
const RECENCY_WINDOWS: Record<SidebarRecencyFilter, number> = {
  '1d': 24 * 60 * 60,
  '2d': 2 * 24 * 60 * 60
}

/**
 * Narrow to sessions worked on within any of the selected windows — "1 day"
 * is the last 24 hours, "2 day" the last 48. Age comes from `sessionRecency`
 * (last_active, else started_at), the same clock the rows sort and label by,
 * so a recency filter and the date dividers agree about what "yesterday"
 * means.
 *
 * Selection is a union, like every other sidebar filter: picking both windows
 * is just the 48-hour one. Fails closed only for a session with no timestamps
 * at all — nothing there can prove it falls inside a window.
 */
export function sessionMatchesRecencyFilter(
  session: SessionInfo,
  windows: readonly SidebarRecencyFilter[],
  nowSeconds: number
): boolean {
  if (!windows.length) {
    return true
  }

  const recency = sessionRecency(session)

  if (recency <= 0) {
    return false
  }

  const ageSeconds = nowSeconds - recency

  return windows.some(id => ageSeconds <= RECENCY_WINDOWS[id])
}
