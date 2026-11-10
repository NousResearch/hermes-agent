import { describe, expect, it } from 'vitest'

import { formatPercent, formatPercentUsed } from './usage-percent'

describe('formatPercentUsed', () => {
  it('uses the requested locale for percent signs and digits', () => {
    for (const locale of ['ar', 'en', 'de']) {
      expect(formatPercent(0.47, locale)).toBe(new Intl.NumberFormat(locale, { style: 'percent' }).format(0.47))
    }
  })
  it('floors, shows <1% for any sliver, and 100% only when exhausted', () => {
    expect(formatPercentUsed(0)).toBe('0%')
    expect(formatPercentUsed(0.0047)).toBe('<1%')
    expect(formatPercentUsed(1 - 0.99)).toBe('1%')
    expect(formatPercentUsed(0.996)).toBe('99%')
    expect(formatPercentUsed(0.9999)).toBe('99%')
    expect(formatPercentUsed(1)).toBe('100%')
  })
})
