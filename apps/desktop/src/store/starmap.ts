import { atom } from 'nanostores'

import { getStarmapGraph } from '@/hermes'
import type { StarmapGraph } from '@/types/hermes'

// On-demand cache for the star map. The graph scan touches the skills catalog +
// usage ledger + memory files, so we fetch it only when the panel opens (and on
// an explicit refresh), never on a turn boundary.
export const $starmapGraph = atom<StarmapGraph | null>(null)
export const $starmapLoading = atom(false)
export const $starmapError = atom<null | string>(null)

let inflight: Promise<void> | null = null

export async function loadStarmapGraph(force = false): Promise<void> {
  if (inflight) {
    return inflight
  }

  if ($starmapGraph.get() && !force) {
    return
  }

  $starmapLoading.set(true)
  $starmapError.set(null)

  inflight = (async () => {
    try {
      $starmapGraph.set(await getStarmapGraph())
    } catch (err) {
      $starmapError.set(err instanceof Error ? err.message : String(err))
    } finally {
      $starmapLoading.set(false)
      inflight = null
    }
  })()

  return inflight
}

/** Drop the cache so the next open refetches against the now-active profile. */
export function resetStarmapGraph(): void {
  inflight = null
  $starmapGraph.set(null)
  $starmapError.set(null)
}
