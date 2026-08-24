import { describe, expect, it } from 'vitest'

import { contrastRatio, mix } from './color'
import { BUILTIN_THEME_LIST } from './presets'
import type { DesktopTheme, DesktopThemeColors } from './types'

/**
 * Guard the `--ui-text-*` token ramp in styles.css against contrast regressions.
 *
 * Those tokens are derived as `color-mix(in srgb, var(--ui-base) P%, transparent)`,
 * which paints as `base at P% alpha` over the surface beneath. The lowest rungs
 * (tertiary 54%, quaternary 36%) fell below WCAG AA (4.5:1) on light themes and
 * were raised to 68%. This test re-derives the composited color and asserts the
 * informative text levels clear AA on the surfaces where secondary/status text
 * actually renders (app background, sidebar, and card).
 *
 * Scope: first-party themes only. catppuccin, everforest and solarized are
 * marketplace forks whose foregrounds come from upstream (see the comment atop
 * presets.ts - "re-convert marketplace forks from the upstream extension rather
 * than hand-editing hexes"), so we can't fix their contrast here without
 * drifting from upstream. They remain known debt, tracked against #38072.
 */

const TEXT_LEVELS = {
  // `--ui-text-secondary` (74%) drives `--dt-secondary-foreground`.
  secondary: 0.74,
  // `--ui-text-tertiary` (68%) drives the status bar and command-center captions.
  tertiary: 0.68,
  // `--ui-text-quaternary` (68%) drives timestamps and empty-state counts.
  // Intentionally equal to tertiary: at these alphas no gap both keeps a
  // visual step and clears AA, so we trade the finer hierarchy for readable text.
  quaternary: 0.68
} as const

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
  c.card
]

/** A theme's renderable palettes; first-party single-mode themes share one. */
const palettesFor = (theme: DesktopTheme): Array<[string, DesktopThemeColors]> => [
  ['light', theme.colors],
  ...(theme.darkColors && theme.darkColors !== theme.colors
    ? ([['dark', theme.darkColors]] as Array<[string, DesktopThemeColors]>)
    : [])
]

describe('ui-text token contrast', () => {
  for (const theme of BUILTIN_THEME_LIST) {
    if (!FIRST_PARTY_NAMES.has(theme.name)) {
      continue
    }

    for (const [mode, colors] of palettesFor(theme)) {
      describe(`${theme.name} (${mode})`, () => {
        for (const [level, pct] of Object.entries(TEXT_LEVELS)) {
          it(`${level} clears 4.5:1 on every status surface`, () => {
            for (const surface of surfacesFor(colors)) {
              // Re-derive the composited token: base at P% over the surface.
              const token = mix(surface, colors.foreground, pct)
              expect(contrastRatio(token, surface), `${level} on ${surface}`).toBeGreaterThanOrEqual(4.5)
            }
          })
        }
      })
    }
  }
})
