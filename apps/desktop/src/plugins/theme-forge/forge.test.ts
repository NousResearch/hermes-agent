/**
 * Theme Forge — color math + synthesis validation.
 *
 * Vitest port of the standalone `forge-math-test.cjs` harness (74 checks)
 * that validates the disk plugin's theme math. Same synthetic-palette
 * fixtures, same verbatim-contract assertions, plus the structural checks
 * that now come free from TypeScript.
 */

import { describe, expect, it } from 'vitest'

import {
  ansiPalette,
  buildColorsFromPalette,
  contrast,
  deriveSwatches,
  type ForgeMode,
  hexToHsl,
  hexToRgb,
  hslToHex,
  luminance,
  rgbToHex,
  rgbToHsl,
  stripForgePrefix,
  type Swatch,
  synthesize
} from './forge'

const HEX = /^#[0-9a-f]{6}$/i
const REQUIRED = ['background', 'foreground', 'primary'] as const

/** Synthetic swatch factory matching the harness's test-image blocks. */
const mk = (r: number, g: number, b: number, weight: number): Swatch => ({
  hex: rgbToHex(r, g, b),
  hsl: rgbToHsl(r, g, b),
  weight
})

const palette = [
  mk(13, 47, 134, 2000),
  mk(255, 120, 50, 1800),
  mk(255, 224, 196, 1500),
  mk(120, 30, 80, 1400),
  mk(46, 160, 120, 1300),
  mk(240, 200, 60, 1200),
  mk(110, 60, 110, 600) // gradient average
]

const hueDist = (a: number, b: number): number => {
  const d = Math.abs(a * 360 - b * 360) % 360

  return d > 180 ? 360 - d : d
}

describe('buildColorsFromPalette — token validity', () => {
  for (const wantDark of [true, false]) {
    const label = wantDark ? 'dark' : 'light'
    const c = buildColorsFromPalette(palette, wantDark)

    it(`[${label}] all tokens are 6-digit hex`, () => {
      for (const [k, v] of Object.entries(c)) {
        expect(v, k).toMatch(HEX)
      }
    })

    it(`[${label}] required keys present`, () => {
      for (const k of REQUIRED) {
        expect(typeof c[k], k).toBe('string')
      }
    })

    it(`[${label}] bg equals slot-1 swatch VERBATIM`, () => {
      expect(c.background.toLowerCase()).toBe(palette[0].hex.toLowerCase())
    })

    it(`[${label}] fg equals slot-2 swatch VERBATIM`, () => {
      expect(c.foreground.toLowerCase()).toBe(palette[1].hex.toLowerCase())
    })

    it(`[${label}] primary equals first chromatic swatch VERBATIM`, () => {
      expect(c.primary.toLowerCase()).toBe(palette[1].hex.toLowerCase())
    })

    it(`[${label}] mutedFg contrast >= 4.5`, () => {
      expect(contrast(c.mutedForeground, c.background)).toBeGreaterThanOrEqual(4.5)
    })

    it(`[${label}] terminal palette: 16 ANSI slots are hex`, () => {
      const t = ansiPalette(palette, c.background, palette[1].hex)

      const ansiKeys = [
        'black',
        'red',
        'green',
        'yellow',
        'blue',
        'magenta',
        'cyan',
        'white',
        'brightBlack',
        'brightRed',
        'brightGreen',
        'brightYellow',
        'brightBlue',
        'brightMagenta',
        'brightCyan',
        'brightWhite'
      ] as const

      for (const k of ansiKeys) {
        expect(t[k], k).toMatch(HEX)
      }

      expect(typeof t.foreground).toBe('string')
      expect(typeof t.cursor).toBe('string')
    })

    it(`[${label}] terminal fg equals slot-2 swatch VERBATIM`, () => {
      const t = ansiPalette(palette, c.background, palette[1].hex)
      expect(t.foreground?.toLowerCase()).toBe(palette[1].hex.toLowerCase())
    })

    it(`[${label}] ANSI colors visible on term bg`, () => {
      const t = ansiPalette(palette, c.background, palette[1].hex)
      // black (dark bg) and white (light bg) are intentionally near-background
      // — standard terminal behavior — so exclude them from the visibility
      // floor. Luminance-based, NOT mode-based: verbatim backgrounds can be
      // dark even in the light variant, and the near-bg ANSI slot should
      // follow the ACTUAL bg.
      const bgLum = luminance(c.background)

      const ansiKeys = [
        'black',
        'red',
        'green',
        'yellow',
        'blue',
        'magenta',
        'cyan',
        'white',
        'brightBlack',
        'brightRed',
        'brightGreen',
        'brightYellow',
        'brightBlue',
        'brightMagenta',
        'brightCyan',
        'brightWhite'
      ] as const

      const visKeys = ansiKeys.filter(k => !(bgLum < 0.5 && k === 'black') && !(bgLum >= 0.5 && k === 'white'))

      for (const k of visKeys) {
        expect(contrast(t[k]!, c.background), k).toBeGreaterThanOrEqual(1.4)
      }
    })
  }
})

describe('synthesize — full theme shape', () => {
  const meta = { name: 'forge-test', label: 'Forge · test', mode: 'dark' as ForgeMode }
  const theme = synthesize(palette, meta)

  it('passes the isValidTheme bar (name/label/colors + required)', () => {
    expect(theme.name).toBeTruthy()
    expect(theme.label).toBeTruthy()

    for (const k of REQUIRED) {
      expect(typeof theme.colors[k], k).toBe('string')
    }
  })

  it('label carries through', () => {
    expect(theme.label).toBe('Forge · test')
  })

  it('reorder-safe synthesis', () => {
    const reordered = [palette[1], ...palette.slice(0, 1), ...palette.slice(2)]
    const t2 = synthesize(reordered, meta)

    for (const k of REQUIRED) {
      expect(typeof t2.colors[k], k).toBe('string')
    }

    expect(typeof t2.terminal.red).toBe('string')
  })
})

describe('deriveSwatches — v1-era theme recovery', () => {
  const meta = { name: 'forge-test', label: 'test', mode: 'dark' as ForgeMode }
  const theme = synthesize(palette, meta)
  const derived = deriveSwatches(theme)

  it('recovers 4-8 swatches from tokens', () => {
    expect(derived.length).toBeGreaterThanOrEqual(4)
    expect(derived.length).toBeLessThanOrEqual(8)
  })

  it('every swatch has valid hex + hsl', () => {
    for (const s of derived) {
      expect(s.hex).toMatch(HEX)
      expect(typeof s.hsl.h).toBe('number')
    }
  })

  it('resynthesis round-trip', () => {
    const t3 = synthesize(derived, meta)

    for (const k of REQUIRED) {
      expect(typeof t3.colors[k], k).toBe('string')
    }

    expect(typeof t3.terminal.green).toBe('string')
  })
})

describe('HSL ↔ HEX round-trip', () => {
  it('round-trips within tolerance', () => {
    const wheelColors = [mk(13, 47, 134, 5000), mk(255, 120, 50, 4000), mk(46, 160, 120, 3000), mk(240, 235, 220, 2000)]

    for (const c of wheelColors) {
      const back = hslToHex(c.hsl.h, c.hsl.s, c.hsl.l)
      const hsl = hexToHsl(back)
      expect(back).toMatch(HEX)
      expect(hsl).toBeTruthy()
      expect(Math.abs(hsl!.h - c.hsl.h)).toBeLessThan(0.005)
      expect(Math.abs(hsl!.s - c.hsl.s)).toBeLessThan(0.005)
      expect(Math.abs(hsl!.l - c.hsl.l)).toBeLessThan(0.005)
    }
  })
})

describe('slot-1 background control', () => {
  // Build an order where slot 1 is NOT the darkest color — old code would
  // have ignored slot 1 and used the darkest anyway.
  const teal = mk(46, 160, 120, 5000) // mid-tone teal, deliberately not darkest
  const deepBlue = mk(13, 47, 134, 4000)
  const ordered2 = [teal, deepBlue, mk(255, 120, 50, 3000), mk(255, 224, 196, 2000), mk(120, 30, 80, 1000)]
  const themeA = synthesize(ordered2, { name: 'x', label: 'x', mode: 'dark' })
  const bgA = themeA.darkColors.background

  it('dark bg hue follows teal seed', () => {
    expect(hueDist(rgbToHsl(...hexToRgb(bgA)!).h, teal.hsl.h)).toBeLessThan(35)
  })

  it('teal seed lands bg VERBATIM', () => {
    expect(bgA.toLowerCase()).toBe(teal.hex.toLowerCase())
  })

  it('swap changes bg hue', () => {
    const ordered3 = [deepBlue, teal, mk(255, 120, 50, 3000), mk(255, 224, 196, 2000), mk(120, 30, 80, 1000)]
    const themeB = synthesize(ordered3, { name: 'x', label: 'x', mode: 'dark' })
    const bgB = themeB.darkColors.background
    expect(hueDist(rgbToHsl(...hexToRgb(bgB)!).h, deepBlue.hsl.h)).toBeLessThan(35)
    expect(hueDist(rgbToHsl(...hexToRgb(bgA)!).h, rgbToHsl(...hexToRgb(bgB)!).h)).toBeGreaterThan(20)
  })

  it('light bg keeps seed VERBATIM', () => {
    const themeC = synthesize([deepBlue, teal, mk(255, 224, 196, 2000)], { name: 'x', label: 'x', mode: 'light' })
    expect(themeC.colors.background.toLowerCase()).toBe(deepBlue.hex.toLowerCase())
  })

  it('bright seed lands bg VERBATIM (no blending to dark)', () => {
    const themeD = synthesize([mk(255, 224, 196, 5000), teal, deepBlue], { name: 'x', label: 'x', mode: 'dark' })
    expect(themeD.darkColors.background.toLowerCase()).toBe(mk(255, 224, 196, 5000).hex.toLowerCase())
  })
})

describe('slot-2 foreground control', () => {
  // Use distinct swatches so hue swaps are observable. Background is a realistic
  // dark (deepBlue) so the contrast floor is a no-op and the verbatim
  // carry-through is visible. (A dedicated floor suite below checks the
  // disaster case.)
  const deepBlue = mk(13, 47, 134, 4000)
  const softPink = mk(230, 150, 160, 1200)
  const skyBlue = mk(120, 180, 255, 1100)
  const slot2Base = [deepBlue, softPink, skyBlue, mk(255, 120, 50, 3000), mk(120, 30, 80, 2000)]
  const themeFG1 = synthesize(slot2Base, { name: 'x', label: 'x', mode: 'dark' })

  const themeFG2 = synthesize([deepBlue, skyBlue, softPink, mk(255, 120, 50, 3000), mk(120, 30, 80, 2000)], {
    name: 'x',
    label: 'x',
    mode: 'dark'
  })

  it('swapping slot 2 changes dark fg hue', () => {
    const h1 = rgbToHsl(...hexToRgb(themeFG1.darkColors.foreground)!).h
    const h2 = rgbToHsl(...hexToRgb(themeFG2.darkColors.foreground)!).h
    expect(hueDist(h1, h2)).toBeGreaterThan(15)
  })

  it('dark fg equals slot-2 swatch VERBATIM', () => {
    expect(themeFG1.darkColors.foreground.toLowerCase()).toBe(softPink.hex.toLowerCase())
  })

  it('light fg equals slot-2 swatch VERBATIM', () => {
    expect(themeFG1.colors.foreground.toLowerCase()).toBe(softPink.hex.toLowerCase())
  })

  it('fg keeps seed chroma (sat > 0.08)', () => {
    const fg1Sat = rgbToHsl(...hexToRgb(themeFG1.darkColors.foreground)!).s
    const fg2Sat = rgbToHsl(...hexToRgb(themeFG2.darkColors.foreground)!).s
    expect(fg1Sat).toBeGreaterThan(0.08)
    expect(fg2Sat).toBeGreaterThan(0.08)
  })

  it('light seed fg does not collapse to white', () => {
    const lightSeedFG = mk(240, 235, 220, 4000)
    const deepBlue = mk(13, 47, 134, 4000)
    const t = synthesize([deepBlue, lightSeedFG, skyBlue], { name: 'x', label: 'x', mode: 'dark' })
    expect(t.darkColors.foreground.toLowerCase()).not.toBe('#ffffff')
    expect(t.darkColors.foreground.toLowerCase()).toBe(lightSeedFG.hex.toLowerCase())
  })

  it('single swatch still derives fg (different from bg)', () => {
    const themeSingle = synthesize([softPink], { name: 'x', label: 'x', mode: 'dark' })
    expect(typeof themeSingle.darkColors.foreground).toBe('string')
    expect(themeSingle.darkColors.foreground.toLowerCase()).not.toBe(themeSingle.darkColors.background.toLowerCase())
  })
})

describe('terminal fg follows slot 2', () => {
  const softPink = mk(230, 150, 160, 1200)
  const skyBlue = mk(120, 180, 255, 1100)
  const deepBlue = mk(13, 47, 134, 4000)
  const slot2Base = [deepBlue, softPink, skyBlue, mk(255, 120, 50, 3000), mk(120, 30, 80, 2000)]
  const themeFG1 = synthesize(slot2Base, { name: 'x', label: 'x', mode: 'dark' })

  const themeFG2 = synthesize([deepBlue, skyBlue, softPink, mk(255, 120, 50, 3000), mk(120, 30, 80, 2000)], {
    name: 'x',
    label: 'x',
    mode: 'dark'
  })

  const termFG1 = themeFG1.darkTerminal.foreground!
  const termFG2 = themeFG2.darkTerminal.foreground!

  it('not hardcoded white/black', () => {
    expect(termFG1.toLowerCase()).not.toBe('#ffffff')
    expect(termFG1.toLowerCase()).not.toBe('#000000')
  })

  it('follows slot-2 seed hue', () => {
    expect(hueDist(rgbToHsl(...hexToRgb(termFG1)!).h, softPink.hsl.h)).toBeLessThan(40)
  })

  it('swapping slot 2 changes terminal fg', () => {
    expect(termFG1).not.toBe(termFG2)
    expect(hueDist(rgbToHsl(...hexToRgb(termFG1)!).h, rgbToHsl(...hexToRgb(termFG2)!).h)).toBeGreaterThan(15)
  })

  it('terminal fg equals slot-2 swatch VERBATIM', () => {
    expect(termFG1.toLowerCase()).toBe(softPink.hex.toLowerCase())
  })

  it('no slot 2 falls back to readable (valid hex, != bg)', () => {
    const themeSingle = synthesize([softPink], { name: 'x', label: 'x', mode: 'dark' })
    expect(themeSingle.darkTerminal.foreground).toMatch(HEX)
    expect(themeSingle.darkTerminal.foreground!.toLowerCase()).not.toBe(themeSingle.darkColors.background.toLowerCase())
  })
})

describe('contrast floor — tripwire', () => {
  // The disaster the floor exists for: near-black bg + near-black text must be
  // nudged to a readable ratio, while a deliberately readable pair stays verbatim.
  const FLOOR = 3
  const nearBlackBg = mk(8, 8, 10, 5000) // ~#08080a
  const nearBlackFg = mk(12, 12, 14, 2000) // ~#0c0c0e (contrast ~1.1:1 on bg)
  const floorTheme = synthesize([nearBlackBg, nearBlackFg], { name: 'floor', label: 'floor', mode: 'dark' })
  const floorFG = floorTheme.darkColors.foreground

  it('near-black text is nudged readable (>= 3:1)', () => {
    expect(contrast(floorFG, floorTheme.darkColors.background)).toBeGreaterThanOrEqual(FLOOR)
  })

  it('nudged fg is NOT the original (actually changed)', () => {
    expect(floorFG.toLowerCase()).not.toBe(nearBlackFg.hex.toLowerCase())
  })

  it('nudged fg is a valid hex', () => {
    expect(floorFG).toMatch(HEX)
  })

  it('terminal fg follows same floor', () => {
    expect(contrast(floorTheme.darkTerminal.foreground!, floorTheme.darkColors.background)).toBeGreaterThanOrEqual(
      FLOOR
    )
  })

  it('near-white light-mode text nudged readable (>= 3:1)', () => {
    const nearWhiteBg = mk(250, 250, 252, 5000)
    const nearWhiteFg = mk(248, 248, 250, 2000)
    const floorLight = synthesize([nearWhiteBg, nearWhiteFg], { name: 'floor', label: 'floor', mode: 'light' })
    expect(contrast(floorLight.colors.foreground, floorLight.colors.background)).toBeGreaterThanOrEqual(FLOOR)
  })

  it('readable pair stays VERBATIM (floor is no-op)', () => {
    // softPink on deepBlue is ~5.8:1 (> 3) so the floor is a no-op.
    const deepBlue = mk(13, 47, 134, 4000)
    const softPink = mk(230, 150, 160, 1200)
    const t = synthesize([deepBlue, softPink, mk(120, 180, 255, 1100)], { name: 'x', label: 'x', mode: 'dark' })
    expect(t.darkColors.foreground.toLowerCase()).toBe(softPink.hex.toLowerCase())
  })

  it('light seed fg VERBATIM on dark bg (no over-nudge to pure white)', () => {
    const deepBlue = mk(13, 47, 134, 4000)
    const lightSeedFG = mk(240, 235, 220, 4000)
    const t = synthesize([deepBlue, lightSeedFG, mk(120, 180, 255, 1100)], { name: 'x', label: 'x', mode: 'dark' })
    expect(t.darkColors.foreground.toLowerCase()).toBe(lightSeedFG.hex.toLowerCase())
  })
})

describe('stripForgePrefix — sleek naming', () => {
  it('removes auto prefix', () => {
    expect(stripForgePrefix('Forge · Sunset')).toBe('Sunset')
  })

  it('is case/separator tolerant', () => {
    expect(stripForgePrefix('forge•  Aurora')).toBe('Aurora')
  })

  it('intentional Forge names preserved', () => {
    expect(stripForgePrefix('Dark Forge')).toBe('Dark Forge')
    expect(stripForgePrefix('Forge Midnight')).toBe('Forge Midnight')
  })

  it('empty-result fallback keeps original', () => {
    expect(stripForgePrefix('Forge · ')).toBe('Forge · ')
  })
})
