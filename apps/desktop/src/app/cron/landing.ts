/** The schedule the detail shows: the one asked for while it is still listed, else the first listed,
 *  so the detail is never empty while there are schedules. `null` only for an empty list. */
export function pickSchedule<T extends { id: string }>(listed: readonly T[], wanted: null | string): T | null {
  return listed.find(job => job.id === wanted) ?? listed[0] ?? null
}
