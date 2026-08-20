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
  backgroundMaterialFor,
  clampIntensity,
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassActive,
  type GlassMaterial,
  glassMaterialForPicker,
  glassMaterialsFor,
  glassSupportedOn,
  glassSurfaceKeep,
  hudFrostFor,
  normalizeMaterial,
  normalizeMode,
  normalizeScope,
  normalizeState,
  TRANSLUCENCY_CURVE,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_OPACITY_FLOOR,
  type TranslucencyState,
  translucencySupportedOn,
  vibrancyFor,
  windowBackingOptions,
  windowOpacityFor,
  WINDOWS_BACKGROUND_MATERIALS,
  WINDOWS_GLASS_MIN_BUILD
} from './translucency'

/** The linear ramp the curve replaced. Endpoints must still agree with it. */
const legacyOpacity = (intensity: number) => 1 - (intensity / 100) * 0.7

const clear = (intensity: number): TranslucencyState => ({
  intensity,
  fade: 0,
  mode: 'clear',
  material: DEFAULT_GLASS_MATERIAL,
  scope: DEFAULT_GLASS_SCOPE
})

const glass = (intensity: number, material: GlassMaterial = DEFAULT_GLASS_MATERIAL, fade = 0): TranslucencyState => ({
  intensity,
  fade,
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
  it('accepts glass only on a platform that has a native material', () => {
    expect(normalizeMode('glass', true)).toBe('glass')
    expect(normalizeMode('glass', false)).toBe('clear')
  })

  it('honours an explicit choice', () => {
    expect(normalizeMode('clear', true)).toBe('clear')
    expect(normalizeMode('glass', true)).toBe('glass')
  })

  // Glass is pre-selected so the better half of the feature is the one you
  // find, which is free because the intensity still starts at 0.
  it('pre-selects glass when the platform supports it and nothing is recorded', () => {
    expect(normalizeMode(undefined, true)).toBe('glass')
    expect(normalizeMode('acrylic', true)).toBe('glass')
    expect(normalizeMode(42, true)).toBe('glass')
    expect(normalizeMode(undefined, false)).toBe('clear')
  })

  // The one case that must NOT flip: a profile carrying a non-zero intensity
  // with no mode has been rendering as clear since before glass existed, and
  // defaulting it to glass would change a window the user already tuned.
  it('leaves an already-tuned legacy profile on clear', () => {
    expect(normalizeMode(undefined, true, 45)).toBe('clear')
    expect(normalizeMode(undefined, true, 1)).toBe('clear')
    expect(normalizeMode(undefined, true, 0)).toBe('glass')

    // An explicit mode still wins over the legacy heuristic.
    expect(normalizeMode('glass', true, 45)).toBe('glass')
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

  // The tint is painted by the renderer, so the intensity lever must never
  // reach setOpacity under glass — that separation is what keeps text sharp.
  it('ignores the intensity lever entirely in glass mode', () => {
    expect(windowOpacityFor(glass(0))).toBe(1)
    expect(windowOpacityFor(glass(60))).toBe(1)
    expect(windowOpacityFor(glass(100))).toBe(1)
  })

  it('fades a glass window only through its own lever, on the ramp clear uses', () => {
    expect(windowOpacityFor(glass(60, DEFAULT_GLASS_MATERIAL, 0))).toBe(1)
    expect(windowOpacityFor(glass(60, DEFAULT_GLASS_MATERIAL, 40))).toBe(windowOpacityFor(clear(40)))
    expect(windowOpacityFor(glass(0, DEFAULT_GLASS_MATERIAL, 100))).toBe(windowOpacityFor(clear(100)))
  })

  it('leaves fade inert under clear, where the intensity lever already is the opacity', () => {
    expect(windowOpacityFor({ ...clear(40), fade: 100 })).toBe(windowOpacityFor(clear(40)))
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

// The HUD is a transparent window, so its frost has no opaque page to hide
// behind: every state that isn't "frost wanted" has to resolve to no material
// at all, or the band leaves a grey slab hanging over another app.
describe('hudFrostFor', () => {
  it('wears the chosen frost on both platforms while the band is showing', () => {
    expect(hudFrostFor(glass(60, 'header'), true)).toEqual({ vibrancy: 'header', backgroundMaterial: 'mica' })
    expect(hudFrostFor(glass(60, 'under-window'), true)).toEqual({
      vibrancy: 'under-window',
      backgroundMaterial: 'acrylic'
    })
  })

  // The material is the whole window rectangle and nothing on the page can
  // clip it, so a hidden band must mean no frost — this is the veto that keeps
  // idle HUD mode the bar and nothing else.
  it('is off whenever the band is not covering the window', () => {
    expect(hudFrostFor(glass(60, 'header'), false)).toEqual({ vibrancy: null, backgroundMaterial: 'none' })
  })

  // ...and the setting is the other veto: Glass off, or the tint at zero,
  // means the HUD never frosts however engaged the band is.
  it('is off whenever glass itself is off', () => {
    expect(hudFrostFor(clear(60), true)).toEqual({ vibrancy: null, backgroundMaterial: 'none' })
    expect(hudFrostFor(glass(0, 'header'), true)).toEqual({ vibrancy: null, backgroundMaterial: 'none' })
  })

  // Unlike a chat window, which keeps 'sidebar' under its titlebar band in
  // every non-glass state. Pinning this is what stops someone "fixing" the
  // null into a resting material and painting the slab back.
  it('resolves off to no material at all, not to a resting one', () => {
    expect(hudFrostFor(clear(60), true).vibrancy).toBeNull()
    expect(vibrancyFor(clear(60))).toBe('sidebar')
  })

  // The tint is painted by the renderer, exactly as it is for a chat window —
  // dragging it must not re-issue setVibrancy, whose 150ms animation restarts
  // on every call and never lets the material settle.
  it('does not move any native property as the tint slider is dragged', () => {
    for (let intensity = 1; intensity <= 100; intensity += 1) {
      expect(hudFrostFor(glass(intensity, 'popover'), true)).toEqual({
        vibrancy: 'popover',
        backgroundMaterial: 'tabbed'
      })
    }
  })
})

describe('glassSupportedOn', () => {
  it('is on for macOS regardless of kernel version', () => {
    expect(glassSupportedOn('darwin')).toBe(true)
    expect(glassSupportedOn('darwin', '24.6.0')).toBe(true)
  })

  it('is on for Windows 11 22H2 and newer, off for everything older', () => {
    expect(glassSupportedOn('win32', `10.0.${WINDOWS_GLASS_MIN_BUILD}`)).toBe(true)
    expect(glassSupportedOn('win32', `10.0.${WINDOWS_GLASS_MIN_BUILD}.1`)).toBe(true)
    expect(glassSupportedOn('win32', `10.0.${WINDOWS_GLASS_MIN_BUILD - 1}`)).toBe(false)
    expect(glassSupportedOn('win32', '10.0.19045')).toBe(false)
    expect(glassSupportedOn('win32', '10.0')).toBe(false)
    expect(glassSupportedOn('win32', '')).toBe(false)
  })

  it('is off on Linux — Electron has no first-party desktop material there', () => {
    expect(glassSupportedOn('linux', '6.8.0')).toBe(false)
  })
})

describe('backgroundMaterialFor', () => {
  it('is none while glass is off so DWM does not keep drawing under the backing', () => {
    expect(backgroundMaterialFor(glass(0, 'header'))).toBe('none')
    expect(backgroundMaterialFor(clear(60))).toBe('none')
  })

  it('maps the sheer → heavy frost ladder onto acrylic / tabbed / mica', () => {
    expect(backgroundMaterialFor(glass(60, 'under-window'))).toBe('acrylic')
    expect(backgroundMaterialFor(glass(60, 'popover'))).toBe('tabbed')
    expect(backgroundMaterialFor(glass(60, 'titlebar'))).toBe('mica')
  })

  // Windows 11 has three system materials for four rungs, so the two heaviest
  // land on mica. The mapping stays total — a saved 'header' still resolves —
  // and the picker drops the duplicate instead (see glassMaterialsFor).
  it('collapses Glare onto mica with Bright', () => {
    expect(backgroundMaterialFor(glass(60, 'header'))).toBe('mica')
    expect(backgroundMaterialFor(glass(60, 'header'))).toBe(backgroundMaterialFor(glass(60, 'titlebar')))
  })

  it('resolves every shipped rung to a real system material', () => {
    for (const material of GLASS_MATERIALS) {
      expect(WINDOWS_BACKGROUND_MATERIALS, material).toContain(backgroundMaterialFor(glass(60, material)))
    }
  })
})

describe('translucencySupportedOn', () => {
  it('covers the two platforms where setOpacity or a native material exists', () => {
    expect(translucencySupportedOn('darwin')).toBe(true)
    expect(translucencySupportedOn('win32')).toBe(true)
  })

  // Electron documents setOpacity as doing nothing on Linux, and there is no
  // material either — so the setting has no working half to offer there.
  it('is off on Linux, where neither mode does anything', () => {
    expect(translucencySupportedOn('linux')).toBe(false)
    expect(translucencySupportedOn('freebsd')).toBe(false)
  })

  // Win10 loses glass but keeps clear, so the row must survive there.
  it('stays on for a Windows build too old for glass', () => {
    const oldWindows = `10.0.${WINDOWS_GLASS_MIN_BUILD - 1}`

    expect(glassSupportedOn('win32', oldWindows)).toBe(false)
    expect(translucencySupportedOn('win32')).toBe(true)
  })
})

describe('the frost rungs a platform offers', () => {
  it('offers the whole ladder on macOS', () => {
    expect(glassMaterialsFor(false)).toEqual(GLASS_MATERIALS)
  })

  // The census rule, now enforced on Windows too: no two options in the picker
  // may composite to the same thing. Bright and Glare are both mica.
  it('never offers two rungs that render the same Windows backdrop', () => {
    const backdrops = glassMaterialsFor(true).map(material => backgroundMaterialFor(glass(60, material)))

    expect(new Set(backdrops).size).toBe(backdrops.length)
    expect(glassMaterialsFor(true).length).toBeLessThan(GLASS_MATERIALS.length)
  })

  it('keeps every distinct Windows backdrop reachable from the picker', () => {
    const backdrops = new Set(glassMaterialsFor(true).map(material => backgroundMaterialFor(glass(60, material))))
    const reachable = new Set(GLASS_MATERIALS.map(material => backgroundMaterialFor(glass(60, material))))

    expect(backdrops).toEqual(reachable)
  })

  // Settings synced from a Mac carry a rung Windows has no button for. The
  // picker highlights the button that renders the same backdrop rather than
  // showing nothing selected — and does NOT rewrite what the Mac saved.
  it('folds a dropped rung onto the button that looks the same', () => {
    expect(glassMaterialForPicker('header', true)).toBe('titlebar')
    expect(glassMaterialsFor(true)).toContain(glassMaterialForPicker('header', true))
    expect(backgroundMaterialFor(glass(60, glassMaterialForPicker('header', true)))).toBe(
      backgroundMaterialFor(glass(60, 'header'))
    )
  })

  it('leaves every rung alone on macOS and every offered rung alone on Windows', () => {
    for (const material of GLASS_MATERIALS) {
      expect(glassMaterialForPicker(material, false)).toBe(material)
    }

    for (const material of glassMaterialsFor(true)) {
      expect(glassMaterialForPicker(material, true)).toBe(material)
    }
  })
})

describe('normalizeState', () => {
  it('parses a modern payload', () => {
    expect(
      normalizeState({ intensity: 40, fade: 15, mode: 'glass', material: 'header', scope: 'sidebar' }, true)
    ).toEqual({
      intensity: 40,
      fade: 15,
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
      fade: 0,
      mode: 'clear',
      material: DEFAULT_GLASS_MATERIAL,
      scope: DEFAULT_GLASS_SCOPE
    })
  })

  // Fade arrived after glass shipped, so a profile written by the older build
  // has no key for it and must come back unfaded rather than undefined.
  it('defaults a payload written before fade existed to no fade', () => {
    expect(normalizeState({ intensity: 60, mode: 'glass' }, true).fade).toBe(0)
  })

  it('survives junk payloads', () => {
    const base = { intensity: 0, fade: 0, material: DEFAULT_GLASS_MATERIAL, scope: DEFAULT_GLASS_SCOPE }

    // A fresh glass-capable profile lands on glass at zero intensity: selected, but off.
    expect(normalizeState(null, true)).toEqual({ ...base, mode: 'glass' })
    expect(normalizeState('nope', true)).toEqual({ ...base, mode: 'glass' })
    expect(
      normalizeState({ intensity: 'x', fade: 'x', material: 'nope', mode: 'glass', scope: 'nope' }, false)
    ).toEqual({
      ...base,
      mode: 'clear'
    })
  })
})

describe('glassActive', () => {
  it('is on only for glass with nonzero intensity', () => {
    expect(glassActive(glass(60))).toBe(true)
    expect(glassActive(glass(0))).toBe(false)
    expect(glassActive(clear(60))).toBe(false)
  })
})

// The default must be selected-but-off: a fresh glass-capable profile shows
// Glass in the picker while the window itself is untouched until the lever moves.
describe('a fresh profile', () => {
  const fresh = normalizeState(null, true)

  it('pre-selects glass with the feature still off', () => {
    expect(fresh.mode).toBe('glass')
    expect(fresh.intensity).toBe(TRANSLUCENCY_MIN)
    expect(glassActive(fresh)).toBe(false)
  })

  it('leaves the window exactly as it is today', () => {
    expect(windowOpacityFor(fresh)).toBe(1)
    expect(windowBackingOptions(fresh, '#101014')).toEqual({ backgroundColor: '#101014' })
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

  // The one glass drag that reaches main, and it costs what a clear drag costs.
  it('is only the opacity while dragging fade under glass', () => {
    expect(nativeDiff(glass(60, DEFAULT_GLASS_MATERIAL, 40), glass(60, DEFAULT_GLASS_MATERIAL, 41))).toEqual({
      backing: false,
      material: false,
      opacity: true
    })
  })

  it('is the material alone when the frost level changes', () => {
    expect(nativeDiff(glass(60, 'under-window'), glass(60, 'header'))).toEqual({
      backing: false,
      material: true,
      opacity: false
    })
    expect(backgroundMaterialFor(glass(60, 'under-window'))).not.toBe(backgroundMaterialFor(glass(60, 'header')))
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
