import { coarseElapsed } from '@hermes/plugin-sdk'

import type { SidebarRowLabels } from './types'

/** Row age in the sidebar's compact form ("now", "52m", "3h", "18d").
 *  Deliberately the same `coarseElapsed` + suffix pair the session rows
 *  directly above use, so the two lists in one rail don't disagree about how
 *  an age is spelled. Not `relativeTime` — that's the bidirectional Intl form
 *  ("in 14 hr"), which belongs on a scheduled next-run. */
export function rowAge(ms: number, r: SidebarRowLabels): string {
  const { unit, value } = coarseElapsed(Date.now() - ms)

  return unit === 'second' ? r.ageNow : `${value}${unit === 'day' ? r.ageDay : unit === 'hour' ? r.ageHour : r.ageMin}`
}
