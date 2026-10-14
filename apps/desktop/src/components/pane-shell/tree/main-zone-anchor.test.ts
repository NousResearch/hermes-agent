// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import { mainZoneAnchorLeft } from './main-zone-anchor'

describe('mainZoneAnchorLeft', () => {
  it('anchors main-view chrome at the zone, not at the window edge', () => {
    expect(mainZoneAnchorLeft(912, 118)).toBe(912)
  })

  it('clamps just right of the window controls when the zone starts under them', () => {
    // A sidebar collapsed to nothing, or a flipped layout, puts the main zone
    // under the cluster: the affordance may not ride beneath the toggle.
    expect(mainZoneAnchorLeft(0, 118)).toBe(130)
    expect(mainZoneAnchorLeft(94, 118)).toBe(130)
  })

  it('leaves the anchor alone when there is no cluster to clear', () => {
    // Chrome hidden (the HUD window), or measured before the controls mount.
    expect(mainZoneAnchorLeft(0, null)).toBe(0)
    expect(mainZoneAnchorLeft(320, null)).toBe(320)
  })
})
