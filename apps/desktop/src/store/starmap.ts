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

// Bumped by resetStarmapGraph on a profile switch: a load still awaiting the
// previous profile's graph must not publish it, nor clear the newer slot.
let starmapEpoch = 0

export async function loadStarmapGraph(force = false): Promise<void> {
  if (inflight) {
    return inflight
  }

  if ($starmapGraph.get() && !force) {
    return
  }

  $starmapLoading.set(true)
  $starmapError.set(null)

  const epoch = starmapEpoch

  inflight = (async () => {
    try {
      const graph = await getStarmapGraph()

      if (epoch === starmapEpoch) {
        $starmapGraph.set(graph)
      }
    } catch (err) {
      if (epoch === starmapEpoch) {
        $starmapError.set(err instanceof Error ? err.message : String(err))
      }
    } finally {
      // Same epoch means this is still the only flight; a stranded one leaves
      // the newer load's spinner and slot alone.
      if (epoch === starmapEpoch) {
        $starmapLoading.set(false)
        inflight = null
      }
    }
  })()

  return inflight
}

/** Drop one node from the cached graph immediately; return rollback. */
export function evictStarmapNode(id: string): () => void {
  const prev = $starmapGraph.get()

  if (!prev) {
    return () => {}
  }

  const next: StarmapGraph = {
    ...prev,
    nodes: prev.nodes.filter(node => node.id !== id),
    edges: prev.edges.filter(edge => edge.source !== id && edge.target !== id)
  }

  $starmapGraph.set(next)

  return () => $starmapGraph.set(prev)
}

/** Drop the cache (and strand any in-flight load) so the next load scans the new scope. Does not
 *  reload: a gateway switch resets before the new connection is active; callers reload an open
 *  view once the new request scope is published (reloadOpenStarmapGraph). */
export function resetStarmapGraph(): void {
  starmapEpoch += 1
  inflight = null
  $starmapGraph.set(null)
  $starmapError.set(null)
  // The stranded load no longer owns the spinner; a fresh load re-raises it.
  $starmapLoading.set(false)
}

let openStarmapViews = 0

/** Reload a mounted Star Map view after a reset, once requests route to the new scope. A view
 *  loads on mount only, so nothing else would refetch it. */
export function reloadOpenStarmapGraph(): void {
  if (openStarmapViews > 0) {
    void loadStarmapGraph()
  }
}

/** Load the graph for a mounted Star Map view and keep it loaded across profile/gateway resets
 *  while the view stays open. Returns the unmount cleanup. */
export function watchStarmapGraph(): () => void {
  openStarmapViews += 1
  void loadStarmapGraph()

  return () => {
    openStarmapViews -= 1
  }
}
