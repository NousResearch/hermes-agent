import { atom } from 'nanostores'

import { getStarmapGraph, getStarmapGraphMultiProfile } from '@/hermes'
import { persistStringArray, storedStringArray } from '@/lib/storage'
import type { StarmapGraph } from '@/types/hermes'

// On-demand cache for the star map. The graph scan touches the skills catalog +
// usage ledger + memory files, so we fetch it only when the panel opens (and on
// an explicit refresh), never on a turn boundary.
export const $starmapGraph = atom<StarmapGraph | null>(null)
export const $starmapLoading = atom(false)
export const $starmapError = atom<null | string>(null)

// ── Multi-profile selection ─────────────────────────────────────────────────
// Persisted list of profile names to include in the journey map. An empty list
// means "current profile only" (single-profile mode, backwards compatible).
// The first profile in the list is considered the "primary" (checked by default
// on first load).
const SELECTED_PROFILES_STORAGE_KEY = 'hermes.desktop.starmap.selectedProfiles'

export const $starmapSelectedProfiles = atom<string[]>(storedStringArray(SELECTED_PROFILES_STORAGE_KEY))

// Persist on change
$starmapSelectedProfiles.subscribe(value => persistStringArray(SELECTED_PROFILES_STORAGE_KEY, [...value]))

export function setStarmapSelectedProfiles(profiles: string[]): void {
  $starmapSelectedProfiles.set(profiles)
}

export function toggleStarmapProfile(profile: string, enabled: boolean): void {
  const current = $starmapSelectedProfiles.get()
  const next = enabled
    ? [...new Set([...current, profile])]
    : current.filter(p => p !== profile)
  $starmapSelectedProfiles.set(next)
}

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
      const selectedProfiles = $starmapSelectedProfiles.get()

      // Multi-profile mode: fetch merged graph when multiple profiles selected
      if (selectedProfiles.length > 1) {
        $starmapGraph.set(await getStarmapGraphMultiProfile(selectedProfiles))
      } else {
        // Single-profile mode (or empty = current profile)
        const profile = selectedProfiles.length === 1 ? selectedProfiles[0] : undefined
        $starmapGraph.set(await getStarmapGraph(profile))
      }
    } catch (err) {
      $starmapError.set(err instanceof Error ? err.message : String(err))
    } finally {
      $starmapLoading.set(false)
      inflight = null
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

/** Drop the cache so the next open refetches against the now-active profile. */
export function resetStarmapGraph(): void {
  inflight = null
  $starmapGraph.set(null)
  $starmapError.set(null)
}

// ── /recall mode ────────────────────────────────────────────────────────────
//
// `/recall` opens the star map in "recall mode": the search sidebar opens
// focused so the user can find a node and insert its knowledge straight into
// the CURRENT chat's composer (as reviewed reference context). The star map is
// an overlay on the active session, so the insert targets the 'main' composer —
// no session id needs threading. This atom is the one-bit signal the slash
// handler raises and the StarmapView consumes (and clears on close).
export const $starmapRecallMode = atom(false)

export function openStarmapRecall(): void {
  $starmapRecallMode.set(true)
}

export function clearStarmapRecall(): void {
  $starmapRecallMode.set(false)
}
