import { contrastRatio } from '@rabbit/shared/color'
import { describe, expect, it } from 'vitest'

import { hexToOklch, withHue } from './color'
import { githubTheme, rabbitTheme } from './presets'
import { retintTheme } from './retint'
import type { DesktopThemeColors } from './types'

const HUES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

// A retint seed for each hue, at the authored accent's lightness/chroma.
const seedAt = (hue: number) => withHue(rabbitTheme.colors.primary, hue)

const RABBIT_BLUE = '#0053FD'

// The two seeds are the whole point of the fork, and both are load-bearing:
// `#0053FD` is the brand color and passes on the light sidebar, but only 3.6:1
// on the near-black dark one — so dark carries a lifted twin rather than the
// literal brand hex. Anything that re-derives these must keep both legible.
describe('the shipped rabbit accents', () => {
  const cases = [
    { appearance: 'light', colors: rabbitTheme.colors },
    { appearance: 'dark', colors: rabbitTheme.darkColors! }
  ] as const

  it.each(cases)('$appearance clears AA on its own sidebar', ({ colors }) => {
    expect(contrastRatio(colors.primary, colors.sidebarBackground!)).toBeGreaterThanOrEqual(4.5)
  })

  it.each(cases)('$appearance keeps text on the accent readable', ({ colors }) => {
    expect(contrastRatio(colors.primary, colors.primaryForeground)).toBeGreaterThanOrEqual(4.5)
  })

  it('leaves GitHub’s neutrals in place — only the accent family is forked', () => {
    for (const key of ['background', 'foreground', 'card', 'border', 'sidebarBackground'] as const) {
      expect(rabbitTheme.colors[key]).toBe(githubTheme.colors[key])
      expect(rabbitTheme.darkColors![key]).toBe(githubTheme.darkColors![key])
    }
  })
})

describe('retintTheme', () => {
  // The load-bearing property: the mix ratios in retint.ts must be the same
  // ones that produced the shipped palette. If they drift, retinting at the
  // theme's OWN hue stops being a no-op — and this catches it.
  it('is an identity at the theme’s own accent', () => {
    const same = retintTheme(rabbitTheme, rabbitTheme.colors.primary)

    expect(same.colors).toEqual(rabbitTheme.colors)
    expect(same.darkColors).toEqual(rabbitTheme.darkColors)
  })

  it('moves every accent-family slot, in both modes', () => {
    const rose = retintTheme(rabbitTheme, seedAt(350))

    for (const mode of ['colors', 'darkColors'] as const) {
      const before = rabbitTheme[mode]!
      const after = rose[mode]!

      for (const key of [
        'primary',
        'ring',
        'midground',
        'composerRing',
        'accent',
        'secondary',
        'userBubble'
      ] as const) {
        expect(after[key], `${mode}.${key}`).not.toBe(before[key])
      }
    }
  })

  it('keeps the four seed slots locked together', () => {
    const teal = retintTheme(rabbitTheme, seedAt(195)).colors

    expect(teal.ring).toBe(teal.primary)
    expect(teal.midground).toBe(teal.primary)
    expect(teal.composerRing).toBe(teal.primary)
  })

  it('leaves the chrome alone', () => {
    // The neutrals are the app's surface, not its brand. A hue knob that also
    // swung these would make every theme a monochrome wash.
    const violet = retintTheme(rabbitTheme, seedAt(285))

    for (const key of ['background', 'foreground', 'card', 'border', 'muted', 'mutedForeground'] as const) {
      expect(violet.colors[key], key).toBe(rabbitTheme.colors[key])
      expect(violet.darkColors![key], `dark ${key}`).toBe(rabbitTheme.darkColors![key])
    }
  })

  it('holds perceived lightness and chroma while only the hue moves', () => {
    const base = hexToOklch(rabbitTheme.colors.primary)!

    for (const hue of HUES) {
      const seed = hexToOklch(retintTheme(rabbitTheme, seedAt(hue)).colors.primary)!

      expect(Math.abs(seed.l - base.l), `L at ${hue}`).toBeLessThan(0.02)
      // Chroma can only be REDUCED, and only where sRGB can't show it.
      expect(seed.c, `C at ${hue}`).toBeLessThanOrEqual(base.c + 0.005)
    }
  })

  // The accent labels the sidebar in small uppercase text, so a hue that
  // collapses against it ships invisible section headers.
  it('keeps the accent readable on the sidebar at every hue', () => {
    for (const hue of HUES) {
      const t = retintTheme(rabbitTheme, seedAt(hue))

      for (const mode of ['colors', 'darkColors'] as const) {
        const c = t[mode] as DesktopThemeColors
        const ratio = contrastRatio(c.primary, c.sidebarBackground ?? c.background)

        expect(ratio, `${mode} @ ${hue}°`).toBeGreaterThanOrEqual(4.5)
      }
    }
  })

  it('re-picks the foreground that sits on the accent', () => {
    for (const hue of HUES) {
      const c = retintTheme(rabbitTheme, seedAt(hue)).colors

      expect(contrastRatio(c.primary, c.primaryForeground), `on-accent @ ${hue}°`).toBeGreaterThanOrEqual(4.5)
    }
  })

  it('accepts any hex form and ignores junk', () => {
    expect(retintTheme(rabbitTheme, '#0053FD').colors.primary).toBe(retintTheme(rabbitTheme, '0053fd').colors.primary)
    // A half-typed hex from a text input must not blow up the theme.
    expect(retintTheme(rabbitTheme, '#00').colors).toEqual(rabbitTheme.colors)
    expect(retintTheme(rabbitTheme, 'nonsense').colors).toEqual(rabbitTheme.colors)
  })

  // The real motivating case: Rabbit blue is legible on GitHub's light sidebar
  // (5.4:1) but NOT its dark one (3.6:1), so dark has to adapt or ship
  // invisible section headers.
  describe('a seed that only works in one mode', () => {
    const blue = retintTheme(rabbitTheme, RABBIT_BLUE)

    it('keeps the picked color where it already passes', () => {
      expect(blue.colors.primary.toLowerCase()).toBe(RABBIT_BLUE.toLowerCase())
    })

    it('lightens it for the mode where it does not', () => {
      const dark = blue.darkColors!.primary

      expect(dark.toLowerCase()).not.toBe(RABBIT_BLUE.toLowerCase())
      expect(contrastRatio(dark, blue.darkColors!.sidebarBackground!)).toBeGreaterThanOrEqual(4.5)
    })

    it('adapts by lightness, holding the hue — so it still reads as the brand', () => {
      const picked = hexToOklch(RABBIT_BLUE)!
      const adapted = hexToOklch(blue.darkColors!.primary)!

      expect(Math.abs(adapted.h - picked.h)).toBeLessThan(3)
      expect(adapted.l).toBeGreaterThan(picked.l)
      // Chroma may only fall because sRGB cannot SHOW that colorfulness at the
      // higher lightness — `#0053FD`'s C 0.26 is out of gamut once lightened,
      // and the clamp trades it away rather than shifting the hue. What must
      // not happen is the mix-toward-white collapse, which would also drag the
      // hue and leave a pastel; staying well clear of half the original chroma
      // is the line between "same blue, lighter" and "washed out".
      expect(adapted.c).toBeGreaterThan(picked.c * 0.55)
    })
  })

  it('does not brand a slot that never tracked the accent', () => {
    // mono's ring is a neutral gray on purpose.
    const neutralRing = {
      ...rabbitTheme,
      colors: { ...rabbitTheme.colors, ring: '#9a9a9a' },
      darkColors: undefined
    }

    expect(retintTheme(neutralRing, '#8250df').colors.ring).toBe('#9a9a9a')
  })

  // A theme may shade its accent across slots rather than repeating one hex —
  // midnight runs a `#8b80e8` ring under a `#ddd6ff` primary. Both are the
  // same violet; matching on exact equality left the ring behind and produced
  // a half-retinted theme.
  describe('a theme whose accent slots are shades of each other', () => {
    const shaded = {
      ...rabbitTheme,
      colors: { ...rabbitTheme.colors, primary: '#ddd6ff', ring: '#8b80e8', midground: '#8b80e8' },
      darkColors: undefined
    }

    it('moves every slot in the family', () => {
      const teal = retintTheme(shaded, '#0f9b8e')

      expect(teal.colors.ring).not.toBe('#8b80e8')
      expect(Math.abs(hexToOklch(teal.colors.ring)!.h - hexToOklch('#0f9b8e')!.h)).toBeLessThan(3)
    })

    it('keeps each slot at its own lightness, rather than flattening them', () => {
      const teal = retintTheme(shaded, '#0f9b8e')
      const ring = hexToOklch(teal.colors.ring)!

      expect(ring.l).toBeCloseTo(hexToOklch('#8b80e8')!.l, 1)
      expect(ring.l).not.toBeCloseTo(hexToOklch(teal.colors.primary)!.l, 1)
    })
  })
})
