import { describe, expect, it } from 'vitest'

import { contrastRatio, ensureContrast, mix, parseColor, readableOn, relativeLuminance, toHex } from './color'

describe('parseColor', () => {
  it('parses hex6, hex3 and rgb() forms', () => {
    expect(parseColor('#1a2b3c')).toEqual([0x1a, 0x2b, 0x3c])
    expect(parseColor('1a2b3c')).toEqual([0x1a, 0x2b, 0x3c])
    expect(parseColor('#abc')).toEqual([0xaa, 0xbb, 0xcc])
    expect(parseColor('rgb(220,255,220)')).toEqual([220, 255, 220])
    expect(parseColor('rgba(10, 20, 30, 0.5)')).toEqual([10, 20, 30])
  })

  it('rejects garbage', () => {
    expect(parseColor('')).toBeNull()
    expect(parseColor('ansi256(245)')).toBeNull()
    expect(parseColor('#12345')).toBeNull()
  })
})

describe('mix', () => {
  it('lerps in sRGB and round-trips through toHex', () => {
    expect(mix('#000000', '#ffffff', 0.5)).toBe('#808080')
    expect(mix('#ff0000', '#0000ff', 0)).toBe('#ff0000')
    expect(mix('#ff0000', '#0000ff', 1)).toBe('#0000ff')
    expect(toHex(parseColor(mix('#123456', '#654321', 0.3))!)).toBe(mix('#123456', '#654321', 0.3))
  })

  it('passes unparseable inputs through unchanged', () => {
    expect(mix('ansi256(245)', '#ffffff', 0.5)).toBe('ansi256(245)')
    expect(mix('#ff0000', 'nope', 0.5)).toBe('#ff0000')
  })
})

describe('contrast', () => {
  it('measures WCAG ratios at the anchors and refuses to measure garbage', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 0)
    expect(contrastRatio('#ffffff', '#ffffff')).toBeCloseTo(1, 5)
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 5)
    expect(relativeLuminance('#000000')).toBeCloseTo(0, 5)
    expect(relativeLuminance('ansi256(245)')).toBeNull()
    expect(contrastRatio('#ffffff', 'ansi256(245)')).toBeNull()
  })

  // Mid-lightness accents are where a luminance threshold and a measurement
  // disagree: GitHub green #4f9e5e sits just under L=0.5 (threshold → white,
  // 3.29:1) while near-black measures 5.50:1. The contract is "the ink that
  // measures better", so assert against the measurement, not a hardcoded hex.
  it.each([
    ['#4f9e5e', ['#000000', '#ffffff']],
    ['#4f9e5e', ['#161616', '#ffffff']],
    ['#cba6f7', ['#161616', '#ffffff']],
    ['#ffffff', ['#000000', '#ffffff']],
    ['#101014', ['#000000', '#ffffff']]
  ] as const)('readableOn(%s) returns the ink with the higher measured contrast', (bg, inks) => {
    const best = inks.reduce((a, b) => (contrastRatio(bg, b)! > contrastRatio(bg, a)! ? b : a))

    expect(readableOn(bg, inks)).toBe(best)
  })

  it('readableOn defaults to the black/white poles', () => {
    expect(readableOn('#ffffff')).toBe('#000000')
    expect(readableOn('#101014')).toBe('#ffffff')
  })

  it.each([
    ['#FFF8DC', '#ffffff', 3.9],
    ['#4f9e5e', '#ffffff', 4.5],
    ['#3a3a5c', '#101014', 4.5],
    ['#0053fd', '#161616', 4.5],
    ['#cba6f7', '#ffffff', 7]
  ])('ensureContrast(%s on %s) clears %s', (color, bg, min) => {
    const fixed = ensureContrast(color, bg, min)

    expect(contrastRatio(fixed, bg)!).toBeGreaterThanOrEqual(min)
  })

  it('ensureContrast leaves passing and unparseable colors byte-identical', () => {
    expect(ensureContrast('#3D2F13', '#ffffff', 3.9)).toBe('#3D2F13')
    expect(ensureContrast('ansi256(245)', '#ffffff', 3.9)).toBe('ansi256(245)')
  })
})
