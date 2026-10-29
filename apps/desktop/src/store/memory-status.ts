import { atom } from 'nanostores'

/** Bumped whenever something that changes `GET /api/memory`'s answer lands
 *  (config autosave, provider credential save, OAuth completion), so the
 *  Memory summary refetches instead of showing the pre-change readiness. */
export const $memoryStatusRevision = atom(0)

export function refreshMemoryStatus(): void {
  $memoryStatusRevision.set($memoryStatusRevision.get() + 1)
}
