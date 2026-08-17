/**
 * The translucency contract both processes read (`@hermes/shared/translucency`)
 * plus the one piece that needs a BrowserWindow to mean anything: which backing
 * a chat window is born with.
 *
 * The ramp assertions carry over from the clear-mode fix — its endpoints are
 * load-bearing, because a persisted intensity has to keep looking the same
 * across the upgrade that curved the middle of the lever.
 */

import { describe, expect, it } from 'vitest'

import {
  clampIntensity,
  glassActive,
  glassSurfaceKeep,
  normalizeMode,
  normalizeState,
  TRANSLUCENCY_CURVE,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_OPACITY_FLOOR,
  windowBackingOptions,
  windowOpacityFor
} from './translucency'

/** The linear ramp the curve replaced. Endpoints must still agree with it. */
const legacyOpacity = (intensity: number) => 1 - (intensity / 100) * 0.7

const clear = (intensity: number) => ({ intensity, mode: 'clear' as const })
const glass = (intensity: number) => ({ intensity, mode: 'glass' as const })

describe('lever bounds', () => {
  it('keeps the bounds and floor stable so persisted settings survive upgrades', () => {
    expect(TRANSLUCENCY_MIN).toBe(0)
    expect(TRANSLUCENCY_MAX).toBe(100)
    expect(TRANSLUCENCY_OPACITY_FLOOR).toBe(0.3)
  })
})

describe('clampIntensity', () => {
  it('clamps to the lever bounds and rounds to a whole percent', () => {
    expect(clampIntensity(-5)).toBe(TRANSLUCENCY_MIN)
    expect(clampIntensity(0)).toBe(0)
    expect(clampIntensity(49.6)).toBe(50)
    expect(clampIntensity(100)).toBe(TRANSLUCENCY_MAX)
    expect(clampIntensity(250)).toBe(TRANSLUCENCY_MAX)
    expect(clampIntensity('35')).toBe(35)
  })

  it('treats junk as 0 (opaque) rather than letting it reach setOpacity', () => {
    expect(clampIntensity(undefined)).toBe(0)
    expect(clampIntensity(null)).toBe(0)
    expect(clampIntensity(NaN)).toBe(0)
    expect(clampIntensity(Infinity)).toBe(0)
    expect(clampIntensity('glass')).toBe(0)
  })
})

describe('normalizeMode', () => {
  it('accepts glass on macOS only — there is no vibrancy to ride elsewhere', () => {
    expect(normalizeMode('glass', true)).toBe('glass')
    expect(normalizeMode('glass', false)).toBe('clear')
  })

  it('falls back to clear for absent, legacy and junk values', () => {
    expect(normalizeMode(undefined, true)).toBe('clear')
    expect(normalizeMode('clear', true)).toBe('clear')
    expect(normalizeMode('acrylic', true)).toBe('clear')
    expect(normalizeMode(42, true)).toBe('clear')
  })
})

describe('windowOpacityFor', () => {
  it('leaves both endpoints bit-identical to the linear ramp it replaced', () => {
    expect(windowOpacityFor(clear(TRANSLUCENCY_MIN))).toBe(legacyOpacity(0))
    expect(windowOpacityFor(clear(TRANSLUCENCY_MAX))).toBe(legacyOpacity(100))
  })

  it('is fully opaque at 0 and exactly the floor at 100', () => {
    expect(windowOpacityFor(clear(0))).toBe(1)
    expect(windowOpacityFor(clear(100))).toBe(1 - (1 - TRANSLUCENCY_OPACITY_FLOOR))
  })

  it('decreases monotonically and never sinks below the floor', () => {
    let previous = windowOpacityFor(clear(TRANSLUCENCY_MIN))

    for (let intensity = TRANSLUCENCY_MIN + 1; intensity <= TRANSLUCENCY_MAX; intensity += 1) {
      const opacity = windowOpacityFor(clear(intensity))

      expect(opacity, `${intensity} should be more see-through than ${intensity - 1}`).toBeLessThan(previous)
      expect(opacity, `${intensity} should stay at or above the floor`).toBeGreaterThanOrEqual(
        TRANSLUCENCY_OPACITY_FLOOR
      )

      previous = opacity
    }
  })

  // The bug the curve fixes: setOpacity fades text too, so anything under ~0.95
  // is where reading gets hard. A linear ramp put that boundary at intensity 7,
  // leaving almost the whole lever unusable.
  it('spends a useful stretch of the lever on readable settings, not its first few percent', () => {
    const readable = []

    for (let intensity = TRANSLUCENCY_MIN; intensity <= TRANSLUCENCY_MAX; intensity += 1) {
      if (windowOpacityFor(clear(intensity)) >= 0.95) {
        readable.push(intensity)
      }
    }

    expect(readable.length, `only ${readable.length} readable settings`).toBeGreaterThan(20)
  })

  it('keeps fine control near the opaque end', () => {
    // Linear gave 0.965 here — a visible jump off "off" on the very first step.
    expect(windowOpacityFor(clear(5))).toBeGreaterThan(0.99)
    expect(windowOpacityFor(clear(1))).toBeGreaterThan(legacyOpacity(1))
    expect(TRANSLUCENCY_CURVE).toBeGreaterThan(1)
  })

  it('clamps before mapping so corrupt state cannot escape the range', () => {
    expect(windowOpacityFor(clear(-40))).toBe(1)
    expect(windowOpacityFor(clear(240))).toBe(windowOpacityFor(clear(TRANSLUCENCY_MAX)))
  })

  it('never fades the native window in glass mode — the renderer paints that effect', () => {
    expect(windowOpacityFor(glass(0))).toBe(1)
    expect(windowOpacityFor(glass(60))).toBe(1)
    expect(windowOpacityFor(glass(100))).toBe(1)
  })
})

describe('glassSurfaceKeep', () => {
  it('mirrors the clear-mode ramp floor so glass stays matte, never bare desktop', () => {
    expect(glassSurfaceKeep(0)).toBe(100)
    expect(glassSurfaceKeep(50)).toBe(65)
    expect(glassSurfaceKeep(100)).toBeCloseTo(30)
  })

  it('clamps its input like every other consumer of the lever', () => {
    expect(glassSurfaceKeep(-40)).toBe(100)
    expect(glassSurfaceKeep(240)).toBeCloseTo(30)
  })
})

describe('normalizeState', () => {
  it('parses a modern payload', () => {
    expect(normalizeState({ intensity: 40, mode: 'glass' }, true)).toEqual({ intensity: 40, mode: 'glass' })
  })

  // The migration contract: a pre-glass translucency.json is intensity-only and
  // always meant clear. It must NOT silently become glass on update.
  it('keeps a legacy intensity-only payload on clear', () => {
    expect(normalizeState({ intensity: 70 }, true)).toEqual({ intensity: 70, mode: 'clear' })
  })

  it('survives junk payloads', () => {
    expect(normalizeState(null, true)).toEqual({ intensity: 0, mode: 'clear' })
    expect(normalizeState('nope', true)).toEqual({ intensity: 0, mode: 'clear' })
    expect(normalizeState({ intensity: 'x', mode: 'glass' }, false)).toEqual({ intensity: 0, mode: 'clear' })
  })
})

describe('glassActive', () => {
  it('is on only for glass with nonzero intensity', () => {
    expect(glassActive(glass(60))).toBe(true)
    expect(glassActive(glass(0))).toBe(false)
    expect(glassActive(clear(60))).toBe(false)
  })
})

describe('windowBackingOptions', () => {
  // The cold-launch bug: `backgroundColor: '#00000000'` on a non-transparent
  // window is silently treated as OPAQUE, so a window born while glass was
  // persisted blocked the vibrancy material no matter how clear the page went.
  // Omitting the key entirely is the only shape that works, and a runtime
  // setBackgroundColor fixup is lost in a fresh window's first seconds.
  it('omits backgroundColor entirely while glass is active', () => {
    expect(windowBackingOptions(glass(60), '#111111')).toEqual({})
    expect('backgroundColor' in windowBackingOptions(glass(60), '#111111')).toBe(false)
  })

  it('keeps the themed anti-flash backing in every other state', () => {
    expect(windowBackingOptions(glass(0), '#111111')).toEqual({ backgroundColor: '#111111' })
    expect(windowBackingOptions(clear(60), '#111111')).toEqual({ backgroundColor: '#111111' })
    expect(windowBackingOptions(clear(0), '#f7f7f7')).toEqual({ backgroundColor: '#f7f7f7' })
  })
})
