import { atom } from 'nanostores'

/** The schedule this window's Schedules page last showed: in memory only, so every window keeps its
 *  own and a fresh window opens on the first schedule (as the Agents page does with agents). */
export const $lastViewedSchedule = atom<null | string>(null)

export function rememberViewedSchedule(id: string) {
  if ($lastViewedSchedule.get() !== id) {
    $lastViewedSchedule.set(id)
  }
}

/** Where a newly opened Schedules page starts: the redesign reopens the schedule this window showed
 *  last (its Agents page does the same); classic starts on the first row. */
export function initialScheduleSelection(redesign: boolean): null | string {
  return redesign ? $lastViewedSchedule.get() : null
}

/** The schedule the stage shows: the one asked for while it is still listed, else the first listed,
 *  so the detail is never empty while there are schedules. `null` only for an empty list. */
export function pickSchedule<T extends { id: string }>(listed: readonly T[], wanted: null | string): T | null {
  return listed.find(job => job.id === wanted) ?? listed[0] ?? null
}
