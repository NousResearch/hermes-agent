// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import { BACK_CONTROL_INSET, mainZoneAnchorLeft } from './main-zone-anchor'

describe('mainZoneAnchorLeft', () => {
  it('anchors main-view chrome at the zone, inset by the chrome gutter', () => {
    // The inset is what keeps the way back out of a page off the sidebar.
    expect(mainZoneAnchorLeft(912, 118)).toBe(912 + BACK_CONTROL_INSET)
  })

  it('clamps just right of the window controls when the zone starts under them', () => {
    // A sidebar collapsed to nothing, or a flipped layout, puts the main zone
    // under the cluster: the affordance may not ride beneath the toggle. No
    // gutter inset here — the sidebar is beneath the cluster, not beside it.
    expect(mainZoneAnchorLeft(0, 118)).toBe(130)
    expect(mainZoneAnchorLeft(94, 118)).toBe(130)
  })

  it('insets the anchor when there is no cluster to clear', () => {
    // Chrome hidden (the HUD window), or measured before the controls mount.
    expect(mainZoneAnchorLeft(0, null)).toBe(BACK_CONTROL_INSET)
    expect(mainZoneAnchorLeft(320, null)).toBe(320 + BACK_CONTROL_INSET)
  })
})
