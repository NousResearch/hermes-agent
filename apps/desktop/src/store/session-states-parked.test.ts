// #77311: a PARKED session tile (pane-lifecycle unmounts it) must not pin its
// transcript — the warm cache treats it as unreferenced and evicts it, and the
// tile's resume path re-hydrates on unpark. In-flight (busy) states are never
// evicted by the cache regardless.
import { describe, expect, it } from 'vitest'

import { $parkedTileStoredIds, setZoneParkedTiles } from './session-states'

describe('parked session tiles', () => {
  it('unions parked tiles across zones and clears a zone on unmount', () => {
    setZoneParkedTiles('zone-a', ['s1', 's2'])
    setZoneParkedTiles('zone-b', ['s3'])
    expect([...$parkedTileStoredIds.get()].sort()).toEqual(['s1', 's2', 's3'])

    const before = $parkedTileStoredIds.get()
    setZoneParkedTiles('zone-b', ['s3'])
    expect($parkedTileStoredIds.get()).toBe(before) // no churn on an identical report

    setZoneParkedTiles('zone-a', [])
    expect([...$parkedTileStoredIds.get()]).toEqual(['s3'])
    setZoneParkedTiles('zone-b', [])
    expect($parkedTileStoredIds.get().size).toBe(0)
  })
})
