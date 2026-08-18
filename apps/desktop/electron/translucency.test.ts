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
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassActive,
  type GlassMaterial,
  glassSurfaceKeep,
  normalizeMaterial,
  normalizeMode,
  normalizeScope,
  normalizeState,
  TRANSLUCENCY_CURVE,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_OPACITY_FLOOR,
  type TranslucencyState,
  vibrancyFor,
  windowBackingOptions,
  windowOpacityFor
} from './translucency'

/** The linear ramp the curve replaced. Endpoints must still agree with it. */
const legacyOpacity = (intensity: number) => 1 - (intensity / 100) * 0.7

const clear = (intensity: number): TranslucencyState => ({
  intensity,
  mode: 'clear',
  material: DEFAULT_GLASS_MATERIAL,
  scope: DEFAULT_GLASS_SCOPE
})

const glass = (intensity: number, material: GlassMaterial = DEFAULT_GLASS_MATERIAL): TranslucencyState => ({
  intensity,
  mode: 'glass',
  material,
  scope: DEFAULT_GLASS_SCOPE
})

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
  // Full range on purpose: the top of the lever is bare, untinted blur, so the
  // slider spans opaque theme -> clean glass. Text and cards keep their own
  // opaque tokens, which is what makes 100% usable rather than unreadable.
  it('runs linear to zero so the top of the lever is untinted glass', () => {
    expect(glassSurfaceKeep(0)).toBe(100)
    expect(glassSurfaceKeep(50)).toBe(50)
    expect(glassSurfaceKeep(100)).toBe(0)
  })

  it('clamps its input like every other consumer of the lever', () => {
    expect(glassSurfaceKeep(-40)).toBe(100)
    expect(glassSurfaceKeep(240)).toBe(0)
  })
})

describe('normalizeMaterial / normalizeScope', () => {
  it('accepts every shipped material and area', () => {
    for (const material of GLASS_MATERIALS) {
      expect(normalizeMaterial(material)).toBe(material)
    }

    for (const scope of GLASS_SCOPES) {
      expect(normalizeScope(scope)).toBe(scope)
    }
  })

  // The picker was rebuilt from a material census precisely because several
  // NSVisualEffectView materials composite identically; a value that isn't in
  // the shipped ladder must not reach setVibrancy.
  it('falls back for junk, and for materials deliberately left out of the ladder', () => {
    expect(normalizeMaterial('sidebar')).toBe(DEFAULT_GLASS_MATERIAL)
    expect(normalizeMaterial('hud')).toBe(DEFAULT_GLASS_MATERIAL)
    expect(normalizeMaterial(undefined)).toBe(DEFAULT_GLASS_MATERIAL)
    expect(normalizeMaterial(7)).toBe(DEFAULT_GLASS_MATERIAL)
    expect(normalizeScope('rail')).toBe(DEFAULT_GLASS_SCOPE)
    expect(normalizeScope(null)).toBe(DEFAULT_GLASS_SCOPE)
  })
})

describe('vibrancyFor', () => {
  it('uses the chosen frost material only while glass is actually on', () => {
    expect(vibrancyFor(glass(60, 'header'))).toBe('header')
    expect(vibrancyFor(glass(60, 'popover'))).toBe('popover')
  })

  // 'sidebar' is what the titlebar band was designed against, so every
  // non-glass state has to land back on it rather than keeping a stale pick.
  it('falls back to the long-standing sidebar material otherwise', () => {
    expect(vibrancyFor(glass(0, 'header'))).toBe('sidebar')
    expect(vibrancyFor(clear(60))).toBe('sidebar')
  })
})

describe('normalizeState', () => {
  it('parses a modern payload', () => {
    expect(normalizeState({ intensity: 40, mode: 'glass', material: 'header', scope: 'sidebar' }, true)).toEqual({
      intensity: 40,
      mode: 'glass',
      material: 'header',
      scope: 'sidebar'
    })
  })

  // The migration contract: a pre-glass translucency.json is intensity-only and
  // always meant clear. It must NOT silently become glass on update.
  it('keeps a legacy intensity-only payload on clear', () => {
    expect(normalizeState({ intensity: 70 }, true)).toEqual({
      intensity: 70,
      mode: 'clear',
      material: DEFAULT_GLASS_MATERIAL,
      scope: DEFAULT_GLASS_SCOPE
    })
  })

  it('survives junk payloads', () => {
    const fallback = { intensity: 0, material: DEFAULT_GLASS_MATERIAL, mode: 'clear', scope: DEFAULT_GLASS_SCOPE }

    expect(normalizeState(null, true)).toEqual(fallback)
    expect(normalizeState('nope', true)).toEqual(fallback)
    expect(normalizeState({ intensity: 'x', material: 'nope', mode: 'glass', scope: 'nope' }, false)).toEqual(fallback)
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

// The jank fix, as a contract: dragging the intensity slider under glass must
// not touch ANY native property. Each tick used to re-issue setVibrancy, whose
// 150ms animation then restarted before macOS could settle the material —
// which is both the stutter and the reason the frost levels read alike.
describe('what an update actually changes natively', () => {
  const nativeDiff = (previous: TranslucencyState, next: TranslucencyState) => ({
    backing: glassActive(previous) !== glassActive(next),
    material: vibrancyFor(previous) !== vibrancyFor(next),
    opacity: windowOpacityFor(previous) !== windowOpacityFor(next)
  })

  it('is nothing at all while dragging intensity under glass', () => {
    for (let intensity = 41; intensity <= 100; intensity += 1) {
      expect(nativeDiff(glass(intensity - 1), glass(intensity)), `at ${intensity}`).toEqual({
        backing: false,
        material: false,
        opacity: false
      })
    }
  })

  it('is only the opacity while dragging intensity under clear', () => {
    expect(nativeDiff(clear(40), clear(41))).toEqual({ backing: false, material: false, opacity: true })
  })

  it('is the material alone when the frost level changes', () => {
    expect(nativeDiff(glass(60, 'under-window'), glass(60, 'header'))).toEqual({
      backing: false,
      material: true,
      opacity: false
    })
  })

  // Crossing zero flips glass on/off, which is exactly when the backing has to
  // move — the one intensity change that is NOT free.
  it('is the backing and material when glass crosses zero', () => {
    expect(nativeDiff(glass(0), glass(1))).toEqual({ backing: true, material: true, opacity: false })
    expect(nativeDiff(glass(1), glass(0))).toEqual({ backing: true, material: true, opacity: false })
  })

  it('is everything when switching between the two modes', () => {
    expect(nativeDiff(clear(60), glass(60))).toEqual({ backing: true, material: true, opacity: true })
  })
})
