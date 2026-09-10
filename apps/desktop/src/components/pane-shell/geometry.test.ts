import { describe, expect, it } from 'vitest'

import { TITLEBAR_DRAG_FILL_INSET, titlebarDragFillStart } from './geometry'

describe('titlebarDragFillStart', () => {
  it('is 0 when the zone already starts past the control cluster', () => {
    expect(titlebarDragFillStart(98, 96, 260)).toBe(0)
  })

  it('cuts the fill so it starts after the cluster when the zone slides under it', () => {
    expect(titlebarDragFillStart(98, 96, 28)).toBe(166)
  })
})

describe('TITLEBAR_DRAG_FILL_INSET', () => {
  it('references the published titlebar and workspace CSS vars', () => {
    expect(TITLEBAR_DRAG_FILL_INSET).toContain('--titlebar-controls-left')
    expect(TITLEBAR_DRAG_FILL_INSET).toContain('--titlebar-controls-width')
    expect(TITLEBAR_DRAG_FILL_INSET).toContain('--workspace-left')
  })
})
