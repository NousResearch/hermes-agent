import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { PORTAL_ACCENTS, type PortalAccent } from '@/store/portal-accent'
import { portalTheme } from '@/themes/portal'

// The shipped stylesheet's text (vitest does not load CSS modules).
const raw = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'raw.css'), 'utf8')

// "Make it yours" lets the redesign's brand gesture wear eight colours (store/portal-accent.ts). Each
// must keep every contrast duty the blue has, in light and in dark (WCAG AA, 4.5:1 for text): the
// on-brand ink on the fill, the brand ink on every panel and on a brand selection, and the text on a
// brand selection. A bright fill meets the first with a dark on-brand ink, not by being dimmed. Computed from the values the stylesheets ship: the primitives in raw.css (and the
// accent each `data-portal-accent` block selects) and the Hermes Portal skin's seeds.

const AA = 4.5

/** `--portal-raw-*` literals of raw.css, and the accent pair each accent block selects. */
function readPrimitives() {
  const literals = new Map<string, string>()

  for (const [, name, value] of raw.matchAll(/(--portal-raw-[\w-]+):\s*(#[0-9a-f]{3,6})\s*;/gi)) {
    literals.set(name!, value!.toLowerCase())
  }

  const resolve = (token: string) => {
    const value = literals.get(token)

    if (!value) {
      throw new Error(`raw.css has no literal for ${token}`)
    }

    return value
  }

  const named = (block: string, token: string) => {
    const name = new RegExp(`${token}:\\s*var\\((--portal-raw-[\\w-]+)\\)`).exec(block)?.[1]

    if (!name) {
      throw new Error(`an accent block names no ${token}`)
    }

    return resolve(name)
  }

  const set = (block: string) => ({
    fill: named(block, '--portal-raw-accent'),
    on: named(block, '--portal-raw-accent-on'),
    inkLight: named(block, '--portal-raw-accent-ink-light'),
    inkDark: named(block, '--portal-raw-accent-ink-dark')
  })

  const rootBlock = /:root\[data-shell='portal'\] \{([^}]*)\}/.exec(raw)?.[1] ?? ''
  const accents = {} as Record<PortalAccent, ReturnType<typeof set>>

  for (const accent of PORTAL_ACCENTS) {
    const block =
      accent === 'blue'
        ? rootBlock
        : (new RegExp(`\\[data-portal-accent='${accent}'\\] \\{([^}]*)\\}`).exec(raw)?.[1] ?? '')

    accents[accent] = set(block)
  }

  return { accents, resolve }
}

type Rgb = [number, number, number]

const rgb = (hex: string): Rgb => {
  const full = hex.length === 4 ? `#${[...hex.slice(1)].map(c => c + c).join('')}` : hex

  return [1, 3, 5].map(i => parseInt(full.slice(i, i + 2), 16)) as Rgb
}

/** `color-mix(in srgb, top p, bottom)`: an alpha-composited ink is the same mix. */
const mix = (top: Rgb, bottom: Rgb, p: number): Rgb => top.map((c, i) => c * p + bottom[i]! * (1 - p)) as Rgb

const luminance = (color: Rgb) => {
  const [r, g, b] = color.map(c => {
    const v = c / 255

    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  })

  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!
}

const contrast = (a: Rgb, b: Rgb) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)

  return (hi! + 0.05) / (lo! + 0.05)
}

const { accents, resolve } = readPrimitives()
const pureWhite = rgb(resolve('--portal-raw-pure-white'))
const black = rgb(resolve('--portal-raw-black'))
const blue = rgb(resolve('--portal-raw-blue'))
const dark = portalTheme.darkColors!
const light = portalTheme.colors

// The panels a brand mark sits on. Dark: the skin's window, stage, card and floating seeds, and the
// conversation stage as color.css flattens it (white .05 / black .85 / blue .60 over the window).
const darkWindow = rgb('#0a0a1a')

const darkPanels: Rgb[] = [
  rgb(dark.background),
  rgb(dark.card),
  rgb(dark.popover),
  rgb(dark.sidebarBackground!),
  darkWindow,
  mix(pureWhite, mix(black, mix(blue, darkWindow, 0.6), 0.85), 0.05)
]

// Light: the canvas, its window, white veils and cards.
const lightPanels: Rgb[] = [rgb(light.background), rgb(light.card), rgb('#ededed'), pureWhite]

describe.each(PORTAL_ACCENTS)('the %s accent', accent => {
  const fill = rgb(accents[accent].fill)
  const on = rgb(accents[accent].on)
  const inkLight = rgb(accents[accent].inkLight)
  const inkDark = rgb(accents[accent].inkDark)

  it('keeps the on-brand ink readable on its fill', () => {
    expect(contrast(on, fill)).toBeGreaterThanOrEqual(AA)
  })

  it('in light: the brand ink, and the text on a brand selection, read on every panel', () => {
    const ink = rgb(light.foreground)
    const secondary = (panel: Rgb) => mix(ink, panel, 0.8)

    for (const panel of lightPanels) {
      const selected = mix(fill, panel, 0.12)

      expect(contrast(inkLight, panel)).toBeGreaterThanOrEqual(AA)
      expect(contrast(inkLight, selected)).toBeGreaterThanOrEqual(AA)
      expect(contrast(ink, selected)).toBeGreaterThanOrEqual(AA)
      expect(contrast(secondary(selected), selected)).toBeGreaterThanOrEqual(AA)
    }
  })

  it('in dark: the brand ink, and the text on a brand selection, read on every panel', () => {
    const ink = rgb(dark.foreground)
    const secondary = (panel: Rgb) => mix(ink, panel, 0.7)

    for (const panel of darkPanels) {
      const selected = mix(fill, panel, 0.25)

      expect(contrast(inkDark, panel)).toBeGreaterThanOrEqual(AA)
      expect(contrast(inkDark, selected)).toBeGreaterThanOrEqual(AA)
      expect(contrast(ink, selected)).toBeGreaterThanOrEqual(AA)
      expect(contrast(secondary(selected), selected)).toBeGreaterThanOrEqual(AA)
    }
  })
})
