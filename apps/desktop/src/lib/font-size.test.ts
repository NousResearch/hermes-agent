import { afterEach, describe, expect, it } from 'vitest'

import { applyConfiguredFontSize, normalizeUiFontSize } from './font-size'

describe('normalizeUiFontSize', () => {
  it('keeps 0 as default', () => {
    expect(normalizeUiFontSize(0)).toBe(0)
    expect(normalizeUiFontSize('')).toBe(0)
  })

  it('clamps positive values to the supported UI range', () => {
    expect(normalizeUiFontSize(9)).toBe(10)
    expect(normalizeUiFontSize(18)).toBe(18)
    expect(normalizeUiFontSize('20')).toBe(20)
    expect(normalizeUiFontSize(100)).toBe(32)
  })
})

describe('applyConfiguredFontSize', () => {
  afterEach(() => {
    document.documentElement.style.removeProperty('font-size')
  })

  it('applies a configured root font size', () => {
    expect(applyConfiguredFontSize({ display: { font_size: 18 } })).toBe(18)
    expect(document.documentElement.style.fontSize).toBe('18px')
  })

  it('removes the inline override for the default value', () => {
    document.documentElement.style.fontSize = '20px'
    expect(applyConfiguredFontSize({ display: { font_size: 0 } })).toBe(0)
    expect(document.documentElement.style.fontSize).toBe('')
  })
})
