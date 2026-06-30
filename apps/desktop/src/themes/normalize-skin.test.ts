import { beforeEach, describe, expect, it } from 'vitest'

import { storedString, storedStringRecord, writeKey } from '@/lib/storage'

import { applyTheme, deriveTheme, normalizeSkin, PROFILE_SKINS_KEY, SKIN_KEY, skinPref } from './context'
import { DEFAULT_SKIN_NAME, resolveAlias } from './presets'

// jsdom in this environment does not provide a working localStorage.clear()
// (pre-existing baseline issue). Install a full mock so storage-backed code
// paths (normalizeSkin write-back, skinPref.assign/resolve) work in tests.
function installStorageMock() {
  let store: Record<string, string> = {}

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      clear: () => {
        store = {}
      },
      getItem: (key: string) => store[key] ?? null,
      key: (index: number) => Object.keys(store)[index] ?? null,
      removeItem: (key: string) => {
        delete store[key]
      },
      setItem: (key: string, value: string) => {
        store[key] = String(value)
      }
    }
  })
}

describe('resolveAlias', () => {
  it('maps bubblegum-pink to bubblegum', () => {
    expect(resolveAlias('bubblegum-pink')).toBe('bubblegum')
  })

  it('maps nous to bubblegum', () => {
    expect(resolveAlias('nous')).toBe('bubblegum')
  })

  it('maps nous-light to bubblegum', () => {
    expect(resolveAlias('nous-light')).toBe('bubblegum')
  })

  it('maps default to bubblegum', () => {
    expect(resolveAlias('default')).toBe('bubblegum')
  })

  it('maps gold to bubblegum', () => {
    expect(resolveAlias('gold')).toBe('bubblegum')
  })

  it('maps hermes to bubblegum', () => {
    expect(resolveAlias('hermes')).toBe('bubblegum')
  })

  it('passes through unknown names unchanged', () => {
    expect(resolveAlias('midnight')).toBe('midnight')
  })

  it('returns DEFAULT_SKIN_NAME for null', () => {
    expect(resolveAlias(null)).toBe(DEFAULT_SKIN_NAME)
    expect(resolveAlias(null)).toBe('bubblegum')
  })
})

describe('normalizeSkin', () => {
  beforeEach(() => installStorageMock())

  it('returns the canonical default for null', () => {
    expect(normalizeSkin(null)).toBe('bubblegum')
    expect(normalizeSkin(null)).toBe(DEFAULT_SKIN_NAME)
  })

  it('resolves bubblegum-pink to bubblegum via alias', () => {
    expect(normalizeSkin('bubblegum-pink')).toBe('bubblegum')
  })

  it('resolves nous to bubblegum via retired migration', () => {
    expect(normalizeSkin('nous')).toBe('bubblegum')
  })

  it('resolves nous-light to bubblegum', () => {
    expect(normalizeSkin('nous-light')).toBe('bubblegum')
  })

  it('resolves default to bubblegum', () => {
    expect(normalizeSkin('default')).toBe('bubblegum')
  })

  it('resolves gold to bubblegum', () => {
    expect(normalizeSkin('gold')).toBe('bubblegum')
  })

  it('resolves hermes to bubblegum', () => {
    expect(normalizeSkin('hermes')).toBe('bubblegum')
  })

  it('passes through a current non-bubblegum skin unchanged', () => {
    expect(normalizeSkin('midnight')).toBe('midnight')
  })

  it('passes through ember unchanged', () => {
    expect(normalizeSkin('ember')).toBe('ember')
  })

  it('falls back to bubblegum for an unknown name', () => {
    expect(normalizeSkin('nope')).toBe('bubblegum')
  })

  it('persists canonical name to the global localStorage key on migration', () => {
    expect(normalizeSkin('bubblegum-pink')).toBe('bubblegum')
    expect(storedString(SKIN_KEY)).toBe('bubblegum')
  })

  it('does not write back when the name is already canonical', () => {
    persistCanonical('bubblegum')
    expect(normalizeSkin('bubblegum')).toBe('bubblegum')
    // Value stays bubblegum — no spurious write.
    expect(storedString(SKIN_KEY)).toBe('bubblegum')
  })

  it('does not write back for null input', () => {
    expect(normalizeSkin(null)).toBe('bubblegum')
    expect(storedString(SKIN_KEY)).toBeNull()
  })

  it('migrates legacy entries in the per-profile skins map', () => {
    // Seed the profile map with a legacy name.
    skinPref.assign('work', 'nous')
    expect(storedStringRecord(PROFILE_SKINS_KEY)).toMatchObject({ work: 'nous' })

    // Trigger migration via normalizeSkin.
    expect(normalizeSkin('bubblegum-pink')).toBe('bubblegum')

    // The profile map entry should now hold the canonical name.
    expect(storedStringRecord(PROFILE_SKINS_KEY)).toMatchObject({ work: 'bubblegum' })
  })

  it('leaves valid non-bubblegum profile entries untouched during migration sweep', () => {
    skinPref.assign('personal', 'midnight')
    expect(normalizeSkin('bubblegum-pink')).toBe('bubblegum')
    expect(storedStringRecord(PROFILE_SKINS_KEY)).toMatchObject({ personal: 'midnight' })
  })
})

describe('applyTheme overlay injection', () => {
  beforeEach(() => {
    installStorageMock()
    // Remove any overlay style left by the module-load IIFE or prior tests.
    document.getElementById('hermes-theme-overlay-bubblegum')?.remove()
  })

  it('injects the overlay style for canonical bubblegum in light mode', () => {
    const theme = deriveTheme('bubblegum', 'light')
    applyTheme(theme, 'light')

    const overlay = document.getElementById('hermes-theme-overlay-bubblegum')
    expect(overlay).not.toBeNull()
    expect(overlay?.tagName).toBe('STYLE')
  })

  it('injects the overlay style for canonical bubblegum in dark mode', () => {
    const theme = deriveTheme('bubblegum', 'dark')
    applyTheme(theme, 'dark')

    const overlay = document.getElementById('hermes-theme-overlay-bubblegum')
    expect(overlay).not.toBeNull()
  })

  it('sets data-hermes-theme to the canonical bubblegum name', () => {
    const theme = deriveTheme('bubblegum', 'light')
    applyTheme(theme, 'light')

    expect(document.documentElement.dataset.hermesTheme).toBe('bubblegum')
  })

  it('does not inject the overlay for a non-bubblegum skin', () => {
    const theme = deriveTheme('midnight', 'dark')
    applyTheme(theme, 'dark')

    expect(document.getElementById('hermes-theme-overlay-bubblegum')).toBeNull()
  })

  it('removes the overlay style when switching from bubblegum to another skin', () => {
    // First apply bubblegum — overlay should appear.
    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    expect(document.getElementById('hermes-theme-overlay-bubblegum')).not.toBeNull()

    // Switch to midnight — overlay should be gone.
    applyTheme(deriveTheme('midnight', 'dark'), 'dark')
    expect(document.getElementById('hermes-theme-overlay-bubblegum')).toBeNull()
  })

  it('re-injects the overlay when switching back to bubblegum', () => {
    applyTheme(deriveTheme('midnight', 'dark'), 'dark')
    expect(document.getElementById('hermes-theme-overlay-bubblegum')).toBeNull()

    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    expect(document.getElementById('hermes-theme-overlay-bubblegum')).not.toBeNull()
  })

  it('overlay CSS contains both light and dark mode pink gradient rules', () => {
    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    const overlay = document.getElementById('hermes-theme-overlay-bubblegum')
    const css = overlay?.textContent ?? ''

    // Both modes must have their own selector blocks
    expect(css).toContain('[data-hermes-mode="light"]')
    expect(css).toContain('[data-hermes-mode="dark"]')

    // Light mode: pink gradient (not white)
    expect(css).toContain('#FFF0F8')
    expect(css).toContain('#FFCCE0')

    // Dark mode: dark pink/magenta gradient (not black, not violet)
    expect(css).toContain('#2D0E1B')
    expect(css).toContain('#4D1635')
  })

  it('overlay CSS never contains violet/indigo colors', () => {
    applyTheme(deriveTheme('bubblegum', 'dark'), 'dark')
    const overlay = document.getElementById('hermes-theme-overlay-bubblegum')
    const css = overlay?.textContent ?? ''

    // Forbidden violet/indigo colors that were previously in the overlay
    expect(css).not.toContain('199,125,255')
    expect(css).not.toContain('#C77DFF')
    expect(css).not.toContain('#1A0628')
    expect(css).not.toContain('#250A3A')
    expect(css).not.toContain('#3A0A5A')
  })

  it('overlay CSS includes pink scrollbar, focus ring, and selection rules', () => {
    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    const overlay = document.getElementById('hermes-theme-overlay-bubblegum')
    const css = overlay?.textContent ?? ''

    expect(css).toContain('::-webkit-scrollbar')
    expect(css).toContain('::selection')
    expect(css).toContain('input:focus')
    expect(css).toContain('textarea:focus')
  })

  it('sets data-hermes-mode to dark for dark bubblegum theme', () => {
    applyTheme(deriveTheme('bubblegum', 'dark'), 'dark')
    expect(document.documentElement.dataset.hermesMode).toBe('dark')
  })

  it('sets data-hermes-mode to light for light bubblegum theme', () => {
    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    expect(document.documentElement.dataset.hermesMode).toBe('light')
  })

  it('sets a pink background seed CSS var for dark bubblegum theme', () => {
    applyTheme(deriveTheme('bubblegum', 'dark'), 'dark')
    const seed = document.documentElement.style.getPropertyValue('--theme-background-seed')
    expect(seed).toBe('#5A1F41')
  })

  it('sets a pink background seed CSS var for light bubblegum theme', () => {
    applyTheme(deriveTheme('bubblegum', 'light'), 'light')
    const seed = document.documentElement.style.getPropertyValue('--theme-background-seed')
    expect(seed).toBe('#FFB3DF')
  })
})

// ── Palette hue validation ─────────────────────────────────────────────────
// Directly validates the VAL-THEME-002/003 hue-range requirements by
// converting the seed hex values to HSL and asserting they fall in the
// pink/magenta range, never in the forbidden violet/indigo band [240,300].

function hexToHsl(hex: string): { h: number; s: number; l: number } {
  const r = parseInt(hex.slice(1, 3), 16) / 255
  const g = parseInt(hex.slice(3, 5), 16) / 255
  const b = parseInt(hex.slice(5, 7), 16) / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const delta = max - min
  const l = (max + min) / 2

  if (delta === 0) {
    return { h: 0, s: 0, l }
  }

  const s = l > 0.5 ? delta / (2 - max - min) : delta / (max + min)
  let h: number

  if (max === r) {
    h = 60 * (((g - b) / delta) % 6)
  } else if (max === g) {
    h = 60 * ((b - r) / delta + 2)
  } else {
    h = 60 * ((r - g) / delta + 4)
  }

  return { h: h < 0 ? h + 360 : h, s, l }
}

/** Hue is in the pink/magenta range [300,360] or wrapping to [0,20]. */
function isPinkHue(h: number): boolean {
  return h >= 300 || h <= 20
}

describe('bubblegum palette hue validation', () => {
  describe('dark mode', () => {
    const dark = deriveTheme('bubblegum', 'dark').colors

    it('background is dark pink/magenta (hue in pink range, S >= 0.35, L in [0.12,0.30])', () => {
      const { h, s, l } = hexToHsl(dark.background)
      expect(isPinkHue(h)).toBe(true)
      expect(s).toBeGreaterThanOrEqual(0.35)
      expect(l).toBeGreaterThanOrEqual(0.12)
      expect(l).toBeLessThanOrEqual(0.30)
    })

    it('sidebar background is dark pink (hue in pink range, never violet)', () => {
      const { h } = hexToHsl(dark.sidebarBackground ?? dark.background)
      // isPinkHue asserts h >= 300 or h <= 20, which excludes [240,300]
      expect(isPinkHue(h)).toBe(true)
    })

    it('card is dark pink (hue in pink range, never violet)', () => {
      const { h } = hexToHsl(dark.card)
      expect(isPinkHue(h)).toBe(true)
    })

    it('primary is saturated hot pink (S >= 0.55, hue in pink range)', () => {
      const { h, s } = hexToHsl(dark.primary)
      expect(isPinkHue(h)).toBe(true)
      expect(s).toBeGreaterThanOrEqual(0.55)
    })

    it('ring (focus ring) is pink, not cyan or violet', () => {
      const { h } = hexToHsl(dark.ring)
      expect(isPinkHue(h)).toBe(true)
    })

    it('accent is pink (not gold), satisfying VAL-THEME-003', () => {
      const { h } = hexToHsl(dark.accent)
      expect(isPinkHue(h)).toBe(true)
    })

    it('no surface or accent color has hue in the forbidden violet/indigo band [240,300]', () => {
      const surfaces = [
        dark.background,
        dark.card,
        dark.popover,
        dark.muted,
        dark.sidebarBackground ?? dark.background,
        dark.userBubble ?? dark.popover
      ]

      const accents = [dark.primary, dark.ring, dark.accent, dark.border, dark.midground ?? dark.ring]

      for (const hex of [...surfaces, ...accents]) {
        const { h } = hexToHsl(hex)

        // Hue must not fall in [240, 300] — the violet/indigo band
        if (h >= 240 && h <= 300) {
          throw new Error(`${hex} has violet/indigo hue ${h}`)
        }
      }
    })
  })

  describe('light mode', () => {
    const light = deriveTheme('bubblegum', 'light').colors

    it('background is visibly pink (hue in [320,355], S >= 0.20, L >= 0.85)', () => {
      const { h, s, l } = hexToHsl(light.background)
      expect(h).toBeGreaterThanOrEqual(320)
      expect(h).toBeLessThanOrEqual(355)
      expect(s).toBeGreaterThanOrEqual(0.2)
      expect(l).toBeGreaterThanOrEqual(0.85)
    })

    it('sidebar background is pink (hue in pink range)', () => {
      const { h } = hexToHsl(light.sidebarBackground ?? light.background)
      expect(isPinkHue(h)).toBe(true)
    })

    it('card is pink (hue in pink range)', () => {
      const { h } = hexToHsl(light.card)
      expect(isPinkHue(h)).toBe(true)
    })

    it('primary is saturated pink (S >= 0.55, hue in pink range)', () => {
      const { h, s } = hexToHsl(light.primary)
      expect(isPinkHue(h)).toBe(true)
      expect(s).toBeGreaterThanOrEqual(0.55)
    })

    it('ring (focus ring) is saturated pink (S >= 0.55)', () => {
      const { h, s } = hexToHsl(light.ring)
      expect(isPinkHue(h)).toBe(true)
      expect(s).toBeGreaterThanOrEqual(0.55)
    })

    it('accent is saturated pink (S >= 0.55, hue in pink range)', () => {
      const { h, s } = hexToHsl(light.accent)
      expect(isPinkHue(h)).toBe(true)
      expect(s).toBeGreaterThanOrEqual(0.55)
    })
  })
})

// ── helpers ────────────────────────────────────────────────────────────────

/**
 * Writes a canonical skin name directly to the global localStorage key,
 * bypassing normalizeSkin so tests can verify the no-write-back path.
 */
function persistCanonical(name: string): void {
  writeKey(SKIN_KEY, name)
}
