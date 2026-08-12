import { describe, expect, it } from 'vitest'

import { clampIntensity, normalizeMode, normalizePayload, windowOpacityFor } from './translucency'

describe('clampIntensity', () => {
  it('clamps to 0-100 and rounds', () => {
    expect(clampIntensity(-5)).toBe(0)
    expect(clampIntensity(0)).toBe(0)
    expect(clampIntensity(49.6)).toBe(50)
    expect(clampIntensity(100)).toBe(100)
    expect(clampIntensity(250)).toBe(100)
  })

  it('treats junk as 0 (opaque)', () => {
    expect(clampIntensity(undefined)).toBe(0)
    expect(clampIntensity(NaN)).toBe(0)
    expect(clampIntensity('glass')).toBe(0)
    expect(clampIntensity(null)).toBe(0)
  })
})

describe('normalizeMode', () => {
  it('accepts glass on macOS only', () => {
    expect(normalizeMode('glass', true)).toBe('glass')
    expect(normalizeMode('glass', false)).toBe('clear')
  })

  it('falls back to clear for legacy and junk values', () => {
    expect(normalizeMode(undefined, true)).toBe('clear')
    expect(normalizeMode('clear', true)).toBe('clear')
    expect(normalizeMode('acrylic', true)).toBe('clear')
    expect(normalizeMode(42, true)).toBe('clear')
  })
})

describe('windowOpacityFor', () => {
  it('keeps the legacy clear-mode ramp with its 0.3 floor', () => {
    expect(windowOpacityFor(0, 'clear')).toBe(1)
    expect(windowOpacityFor(50, 'clear')).toBeCloseTo(0.65)
    expect(windowOpacityFor(100, 'clear')).toBeCloseTo(0.3)
  })

  it('never fades the native window in glass mode', () => {
    expect(windowOpacityFor(0, 'glass')).toBe(1)
    expect(windowOpacityFor(60, 'glass')).toBe(1)
    expect(windowOpacityFor(100, 'glass')).toBe(1)
  })
})

describe('normalizePayload', () => {
  it('parses a modern payload', () => {
    expect(normalizePayload({ intensity: 40, mode: 'glass' }, true)).toEqual({ intensity: 40, mode: 'glass' })
  })

  it('parses a legacy intensity-only payload as clear', () => {
    expect(normalizePayload({ intensity: 70 }, true)).toEqual({ intensity: 70, mode: 'clear' })
  })

  it('survives junk payloads', () => {
    expect(normalizePayload(null, true)).toEqual({ intensity: 0, mode: 'clear' })
    expect(normalizePayload('nope', true)).toEqual({ intensity: 0, mode: 'clear' })
    expect(normalizePayload({ intensity: 'x', mode: 'glass' }, false)).toEqual({ intensity: 0, mode: 'clear' })
  })
})
