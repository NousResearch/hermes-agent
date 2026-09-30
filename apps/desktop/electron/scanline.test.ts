import { describe, expect, it } from 'vitest'

import {
  normalizeScanlineScope,
  normalizeScanlineState,
  scanlineBoundsForScope,
  scanlineWindowBounds
} from './scanline'

const display = (x: number, y: number, width: number, height: number, internal?: boolean) => ({
  bounds: { x, y, width, height },
  ...(internal === undefined ? {} : { internal })
})

describe('scanlineWindowBounds', () => {
  it('covers a single display exactly', () => {
    expect(scanlineWindowBounds([display(0, 0, 1920, 1080)])).toEqual({
      height: 1080,
      width: 1920,
      x: 0,
      y: 0
    })
  })

  it('spans the union of a multi-monitor desktop (including negative top-left)', () => {
    // Primary right of a secondary that sits up-and-left (negative y).
    const displays = [
      display(0, 0, 1920, 1080, true),
      display(-1920, -200, 1920, 1080)
    ]

    expect(scanlineWindowBounds(displays)).toEqual({
      x: -1920,
      y: -200,
      width: 3840,
      height: 1280
    })
  })

  it('rounds fractional display bounds to whole pixels', () => {
    expect(scanlineWindowBounds([display(0.4, 0.6, 1920.5, 1080.7)])).toEqual({
      height: 1081,
      width: 1921,
      x: 0,
      y: 1
    })
  })

  it('returns a zero box when there are no displays', () => {
    expect(scanlineWindowBounds([])).toEqual({ height: 0, width: 0, x: 0, y: 0 })
  })
})

describe('normalizeScanlineState', () => {
  it('accepts known states and falls back to hidden for anything else', () => {
    expect(normalizeScanlineState('active')).toBe('active')
    expect(normalizeScanlineState('hidden')).toBe('hidden')
    expect(normalizeScanlineState('nonsense')).toBe('hidden')
    expect(normalizeScanlineState(undefined)).toBe('hidden')
    expect(normalizeScanlineState(42)).toBe('hidden')
  })
})

describe('normalizeScanlineScope', () => {
  it('accepts known scopes and falls back to both', () => {
    expect(normalizeScanlineScope('primary')).toBe('primary')
    expect(normalizeScanlineScope('secondary')).toBe('secondary')
    expect(normalizeScanlineScope('both')).toBe('both')
    expect(normalizeScanlineScope('nonsense')).toBe('both')
    expect(normalizeScanlineScope(undefined)).toBe('both')
  })
})

describe('scanlineWindowBoundsForScope', () => {
  // Layout matching the real rig: primary at origin, secondary to its left.
  const displays = [
    display(0, 0, 2560, 1440, true),
    display(-2560, 9, 2560, 1440)
  ]

  it('scopes to the primary display', () => {
    expect(scanlineBoundsForScope(displays, 'primary')).toEqual({
      x: 0, y: 0, width: 2560, height: 1440
    })
  })

  it('scopes to the secondary display (the non-primary one)', () => {
    expect(scanlineBoundsForScope(displays, 'secondary')).toEqual({
      x: -2560, y: 9, width: 2560, height: 1440
    })
  })

  it('spans the union for both', () => {
    expect(scanlineBoundsForScope(displays, 'both')).toEqual({
      x: -2560, y: 0, width: 5120, height: 1449
    })
  })

  it('falls back to the union when the scope is unknown or no display matches', () => {
    expect(scanlineBoundsForScope([display(0, 0, 100, 100, true)], 'secondary')).toEqual({
      x: 0, y: 0, width: 100, height: 100
    })
    expect(scanlineBoundsForScope([], 'primary')).toEqual({
      x: 0, y: 0, width: 0, height: 0
    })
  })
})

describe('scanlineBoundsForScope with an explicit primary id', () => {
  // The real rig: two external 4K monitors, neither "internal". Electron's
  // `internal` flag means "built-in panel" and is false on a dual-external
  // desktop, so only getPrimaryDisplay().id distinguishes the two.
  const externalPair = [
    { id: 11, bounds: { x: 0, y: 0, width: 2560, height: 1440 } },
    { id: 12, bounds: { x: -2560, y: 9, width: 2560, height: 1440 } }
  ]

  it('scopes to the primary by id when neither display is internal', () => {
    expect(scanlineBoundsForScope(externalPair, 'primary', 11)).toEqual({
      x: 0, y: 0, width: 2560, height: 1440
    })
  })

  it('scopes to the secondary (non-primary-id) display when neither is internal', () => {
    expect(scanlineBoundsForScope(externalPair, 'secondary', 11)).toEqual({
      x: -2560, y: 9, width: 2560, height: 1440
    })
  })

  it('still prefers the id over array position when the primary is listed last', () => {
    expect(scanlineBoundsForScope(externalPair, 'secondary', 12)).toEqual({
      x: 0, y: 0, width: 2560, height: 1440
    })
  })
})
