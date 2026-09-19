import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import { contrastRatio, mix } from './color'
import { BUILTIN_THEME_LIST } from './presets'
import { TEXT_RAMP, TEXT_RAMP_CSS_VARS } from './text-ramp'
import type { DesktopTheme, DesktopThemeColors } from './types'

/**
 * Guard the `--ui-text-*` token ramp in styles.css against contrast regressions.
 *
 * Those tokens are derived as `color-mix(in srgb, var(--ui-base) P%, transparent)`,
 * which paints as `base at P% alpha` over the surface beneath. The lowest rungs
 * (tertiary 54%, quaternary 36%) fell below WCAG AA (4.5:1) on light themes and
 * were raised to 68%. This test re-derives the composited color and asserts the
 * informative text levels clear AA on the surfaces where secondary/status text
 * actually renders (app background, sidebar, card, and popover).
 *
 * Scope: first-party themes only. catppuccin, everforest and solarized are
 * marketplace forks whose foregrounds come from upstream (see the comment atop
 * presets.ts - "re-convert marketplace forks from the upstream extension rather
 * than hand-editing hexes"), so we can't fix their contrast here without
 * drifting from upstream. They remain known debt, tracked against #38072.
 */

const STYLES_CSS = resolve(__dirname, '../styles.css')

// The test lives in src/themes/, styles.css lives in src/. The context.tsx
// seed mapping we also pin (--theme-foreground from colors.foreground) is in
// src/themes/context.tsx.
const CONTEXT_TSX = resolve(__dirname, 'context.tsx')

/** The `--ui-text-*` declarations parsed out of styles.css, e.g. 0.74. */
function cssTextAlphas(): Record<string, number> {
  const css = readFileSync(STYLES_CSS, 'utf8')
  const alphas: Record<string, number> = {}

  for (const [level, cssVar] of Object.entries(TEXT_RAMP_CSS_VARS)) {
    const re = new RegExp(`${cssVar}\\s*:\\s*color-mix\\(in srgb, var\\(--ui-base\\)\\s*([\\d.]+)%,\\s*transparent\\)`)
    const match = css.match(re)
    expect(match, `styles.css is missing the ${cssVar} declaration`).toBeTruthy()
    alphas[level] = Number(match![1]) / 100
  }

  return alphas
}

const FIRST_PARTY_NAMES = new Set([
  'nous',
  'github',
  'nous-alt',
  'midnight',
  'ember',
  'mono',
  'slate',
  'cyberpunk'
])

/** The surfaces secondary/status text renders on, per theme mode. */
const surfacesFor = (c: DesktopThemeColors): string[] => [
  c.background,
  c.sidebarBackground ?? c.background,
  c.card,
  c.popover
]

/** A theme's renderable palettes; first-party single-mode themes share one. */
const palettesFor = (theme: DesktopTheme): Array<[string, DesktopThemeColors]> => [
  ['light', theme.colors],
  ...(theme.darkColors && theme.darkColors !== theme.colors
    ? ([['dark', theme.darkColors]] as Array<[string, DesktopThemeColors]>)
    : [])
]

describe('ui-text token contrast', () => {
  // The CSS ramp and the source-of-truth constant must never drift: if someone
  // tweaks an alpha in styles.css, this fails instead of the guard measuring
  // stale numbers (review: constants-drift risk).
  it('matches the ramp declared in styles.css', () => {
    const cssAlphas = cssTextAlphas()
    for (const [level, expected] of Object.entries(TEXT_RAMP)) {
      expect(cssAlphas[level], `--ui-text-* alpha for ${level} drifted from text-ramp.ts`).toBeCloseTo(expected, 2)
    }
  })

  // The test composites `mix(colors.foreground, pct)` while the CSS mixes
  // `var(--ui-base)`. That is only equivalent while --ui-base resolves to the
  // theme's foreground, so pin the chain here: styles.css must derive --ui-base
  // from --theme-foreground (the seed the theme context sets from colors.foreground).
  it('--ui-base derives from the theme foreground (not a repurposed surface)', () => {
    const css = readFileSync(STYLES_CSS, 'utf8')
    expect(css).toMatch(/--ui-base\s*:\s*var\(--theme-foreground\)/)

    // And the theme context publishes --theme-foreground from the theme's
    // foreground seed, closing the --ui-base -> --theme-foreground ->
    // colors.foreground chain that the contrast composition assumes.
    const context = readFileSync(CONTEXT_TSX, 'utf8')
    expect(context).toMatch(/'--theme-foreground'\s*:\s*c\.foreground/)
  })

  for (const theme of BUILTIN_THEME_LIST) {
    if (!FIRST_PARTY_NAMES.has(theme.name)) {
      continue
    }

    for (const [mode, colors] of palettesFor(theme)) {
      describe(`${theme.name} (${mode})`, () => {
        for (const [level, pct] of Object.entries(TEXT_RAMP)) {
          if (level === 'primary') {
            continue
          }

          it(`${level} clears 4.5:1 on every status surface`, () => {
            for (const surface of surfacesFor(colors)) {
              // Re-derive the composited token: --ui-base at P% over the surface.
              const token = mix(surface, colors.foreground, pct)
              expect(contrastRatio(token, surface), `${level} on ${surface}`).toBeGreaterThanOrEqual(4.5)
            }
          })
        }
      })
    }
  }
})
