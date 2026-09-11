import { describe, expect, it } from 'vitest'

import { titlebarDragFillStart } from './geometry'

describe('titlebarDragFillStart', () => {
  it('is 0 when the zone already starts past the control cluster', () => {
    expect(titlebarDragFillStart(98, 96, 260)).toBe(0)
  })

  it('cuts the fill so it starts after the cluster when the zone slides under it', () => {
    expect(titlebarDragFillStart(98, 96, 28)).toBe(166)
  })

  it('keys on the zone, not the workspace: a tool zone at x=28 beside a workspace at x=692 still gets cut', () => {
    // Regression: the inset was computed from --workspace-left (692 → 0 inset)
    // while the terminal zone itself sat at 28 under the cluster.
    expect(titlebarDragFillStart(98, 24, 28)).toBe(94)
    expect(titlebarDragFillStart(98, 24, 692)).toBe(0)
  })
})
