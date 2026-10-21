import { describe, expect, it } from 'vitest'

import { hasTooltipOverflow } from './tooltip'

describe('tooltip overflow measurement', () => {
  it.each([0, 1, 2])('ignores %ipx of vertical glyph ink or rounding', excess => {
    expect(
      hasTooltipOverflow({ clientWidth: 155, scrollWidth: 155, clientHeight: 13, scrollHeight: 13 + excess })
    ).toBe(false)
  })

  it.each([0, 1, 2])('ignores %ipx of horizontal rounding', excess => {
    expect(
      hasTooltipOverflow({ clientWidth: 155, scrollWidth: 155 + excess, clientHeight: 26, scrollHeight: 26 })
    ).toBe(false)
  })

  it('detects a genuinely clamped extra line', () => {
    expect(hasTooltipOverflow({ clientWidth: 155, scrollWidth: 155, clientHeight: 26, scrollHeight: 39 })).toBe(true)
  })

  it('detects genuine horizontal overflow', () => {
    expect(hasTooltipOverflow({ clientWidth: 155, scrollWidth: 170, clientHeight: 26, scrollHeight: 26 })).toBe(true)
  })
})
