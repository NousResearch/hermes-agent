import { describe, expect, it } from 'vitest'

import { zoomAdjustedGapCss } from './panel-titlebar'

describe('zoomAdjustedGapCss', () => {
  it('keeps the physical gap constant as UI scale increases', () => {
    expect(zoomAdjustedGapCss(12, 1.5)).toBe(8)
  })

  it('falls back to the unscaled gap for invalid zoom factors', () => {
    expect(zoomAdjustedGapCss(12, 0)).toBe(12)
    expect(zoomAdjustedGapCss(12, Number.NaN)).toBe(12)
  })
})
