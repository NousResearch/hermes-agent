import { atom } from 'nanostores'

import {
  type ApiRequestScopeToken,
  captureApiRequestScope,
  getStarmapGraph,
  isApiRequestScopeCurrent,
  onApiRequestScopeChange
} from '@/hermes'
import type { StarmapGraph } from '@/types/hermes'

// On-demand cache for the star map. The graph scan touches the skills catalog +
// usage ledger + memory files, so we fetch it only when the panel opens (and on
// an explicit refresh), never on a turn boundary.
export const $starmapGraph = atom<StarmapGraph | null>(null)
export const $starmapLoading = atom(false)
export const $starmapError = atom<null | string>(null)

interface StarmapFlight {
  owner: ApiRequestScopeToken
  promise: Promise<void>
}

let inflight: StarmapFlight | null = null

export async function loadStarmapGraph(force = false): Promise<void> {
  if (inflight) {
    return inflight.promise
  }

  if ($starmapGraph.get() && !force) {
    return
  }

  $starmapLoading.set(true)
  $starmapError.set(null)

  const owner = captureApiRequestScope()
  const flight: StarmapFlight = { owner, promise: Promise.resolve() }

  inflight = flight

  flight.promise = (async () => {
    try {
      const graph = await getStarmapGraph(owner)

      if (inflight === flight && isApiRequestScopeCurrent(owner)) {
        $starmapGraph.set(graph)
      }
    } catch (err) {
      if (inflight === flight && isApiRequestScopeCurrent(owner)) {
        $starmapError.set(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (inflight === flight) {
        inflight = null

        if (isApiRequestScopeCurrent(owner)) {
          $starmapLoading.set(false)
        }
      }
    }
  })()

  return flight.promise
}

/** Drop one node from the cached graph immediately; return rollback. */
export function evictStarmapNode(id: string, owner = captureApiRequestScope()): () => void {
  if (!isApiRequestScopeCurrent(owner)) {
    return () => {}
  }

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

  return () => {
    if (isApiRequestScopeCurrent(owner)) {
      $starmapGraph.set(prev)
    }
  }
}

/** Drop the cache so the next open refetches against the now-active owner. */
export function resetStarmapGraph(): void {
  inflight = null
  $starmapGraph.set(null)
  $starmapLoading.set(false)
  $starmapError.set(null)
}

// The star map is one foreground cache. A same-named profile on another
// connection is a different owner, so clear synchronously when routing moves.
onApiRequestScopeChange(resetStarmapGraph)
