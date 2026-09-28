// @vitest-environment node
import fs from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { aiEvolutionJarvisTheme } from '@/themes/ai-evolution-jarvis'

// `JarvisDashboard` prints `$jarvisUi.result` on the Pulpit — a field the
// gateway fills with the WHOLE answer. Its panel, ink, scale and overflow
// contract live in styles.css (`[data-jarvis-result-pane]`). Dropping the pane,
// clearing its fill or re-inflating the type has to go red here.

const SRC = dirname(fileURLToPath(import.meta.url))
const sourced = fs.readFileSync(resolve(SRC, '../../styles.css'), 'utf8')
// Comments carry prose (and old measurements), never declarations — strip them
// so a rule's selector is the selector, not the note above it.
const stylesheet = sourced.replace(/\/\*[\s\S]*?\*\//g, '')

type Rule = { selector: string; body: string }

/** Every `selector { body }` block in the sheet — flat, braces never nest. */
const rules: Rule[] = [...stylesheet.matchAll(/([^{}]*)\{([^{}]*)\}/g)]
  .map(([, selector, body]) => ({ selector: selector.replace(/\s+/g, ' ').trim(), body }))
  .filter(rule => rule.selector.length > 0)

const paneRule = rules.find(rule => rule.selector === '[data-jarvis-result-pane]')
const paneChildren = rules.filter(rule => rule.selector.startsWith('[data-jarvis-result-pane] '))

const declaration = (property: string): string =>
  // Anchored at a declaration edge (not inside `--conversation-text-font-size`)
  // or the token would answer for the property it is named after.
  paneRule?.body
    .match(new RegExp(`(?:^|[;{\\s])${property}:\\s*([^;]+);`))?.[1]
    .replace(/\s+/g, ' ')
    .trim() ?? ''

/** The top-level arguments of the first `color-mix()` in `text` — commas
 *  inside a nested call don't split. Same helper as chat-surface-glass.test.ts. */
const colorMixArguments = (text: string): string[] | null => {
  const marker = 'color-mix('
  const start = text.indexOf(marker)

  if (start === -1) {
    return null
  }

  const args: string[] = []
  let current = ''
  let depth = 0

  for (const char of text.slice(start + marker.length)) {
    if (char === '(') {
      depth += 1
    } else if (char === ')') {
      if (depth === 0) {
        break
      }

      depth -= 1
    } else if (char === ',' && depth === 0) {
      args.push(current)
      current = ''

      continue
    }

    current += char
  }

  args.push(current)

  return args
}

/** How much of the wallpaper the fill still covers: the painted stop's
 *  percentage, recursively — 86% of the chrome into `transparent` = 0.86. */
const coverage = (text: string): number => {
  const args = colorMixArguments(text)

  if (!args) {
    return 0
  }

  const first = args[1] ?? ''
  const percentage = first.match(/([\d.]+)%\s*$/)
  const own = percentage ? Number(percentage[1]) / 100 : 1
  const stop = first.replace(/([\d.]+)%\s*$/, '').trim()

  const underneath = stop.includes('color-mix(')
    ? coverage(stop)
    : stop.includes('var(--ui-bg-chrome)') && !stop.includes('transparent')
      ? 1
      : 0

  return own * underneath
}

const remOf = (text: string, property = 'font-size'): number => {
  const value = text.match(new RegExp(`${property}:\\s*([\\d.]+)rem`))?.[1]

  return value === undefined ? Number.NaN : Number(value)
}

type Rgb = { b: number; g: number; r: number }

const hexToRgb = (hex: string): Rgb => {
  const digits = hex.replace('#', '')
  const full = digits.length === 3 ? digits.replace(/./g, char => char + char) : digits

  return {
    r: Number.parseInt(full.slice(0, 2), 16),
    g: Number.parseInt(full.slice(2, 4), 16),
    b: Number.parseInt(full.slice(4, 6), 16)
  }
}

/** `left` at `weight` over `right` — both opaque, srgb weights. */
const mix = (left: Rgb, weight: number, right: Rgb): Rgb => ({
  r: left.r * weight + right.r * (1 - weight),
  g: left.g * weight + right.g * (1 - weight),
  b: left.b * weight + right.b * (1 - weight)
})

/** A colour at `alpha` painted over an opaque backdrop — what the eye sees. */
const over = (color: Rgb, alpha: number, backdrop: Rgb): Rgb => mix(color, alpha, backdrop)

const luminance = ({ r, g, b }: Rgb): number => {
  const channel = (value: number) => {
    const scaled = value / 255

    return scaled <= 0.03928 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4
  }

  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

const contrast = (a: Rgb, b: Rgb): number => {
  const [light, dark] = [luminance(a), luminance(b)].sort((left, right) => right - left)

  return (light + 0.05) / (dark + 0.05)
}

/** The knobs the pane's fill is built from: applyTheme() writes
 *  `--theme-background-seed`/`--theme-foreground` from the active skin, so the
 *  shipped default skin supplies those and the sheet supplies the per-mode mix. */
const fillInputs = (mode: 'dark' | 'light') => {
  const chromeTints = [...stylesheet.matchAll(/--theme-neutral-chrome:\s*(#[0-9a-fA-F]{3,8})/g)].map(
    match => match[1]
  )

  const mixKnobs = [...stylesheet.matchAll(/--theme-mix-chrome:\s*([\d.]+)%/g)].map(match => match[1])

  expect(chromeTints).toHaveLength(2)
  expect(mixKnobs).toHaveLength(2)

  const index = mode === 'light' ? 0 : 1
  const darkColors = aiEvolutionJarvisTheme.darkColors

  // The shipped default skin ships both modes; the pane has to clear AA in
  // whichever one the user runs.
  expect(darkColors).toBeDefined()

  const colors = mode === 'light' ? aiEvolutionJarvisTheme.colors : darkColors!

  return {
    chromeTint: hexToRgb(chromeTints[index]),
    ink: hexToRgb(colors.foreground),
    mixChrome: Number(mixKnobs[index]) / 100,
    seed: hexToRgb(colors.background)
  }
}

describe('desktop result pane in styles.css', () => {
  it('owns a tinted, framed and blurred panel instead of the wallpaper', () => {
    expect(paneRule).toBeDefined()
    // Unscoped: the Pulpit sits on the wallpaper in EVERY translucency mode,
    // not just glass/clear — a mode-scoped rule would leave the reported
    // bare-text surface behind in native mode.
    expect(paneRule?.selector).toBe('[data-jarvis-result-pane]')
    expect(paneRule?.body).toMatch(/background:\s*color-mix\([^;]+var\(--ui-bg-chrome\)\s+(?:9\d|100)%/)
    expect(paneRule?.body).not.toMatch(/background(?:-color)?:\s*transparent/)
    expect(declaration('border')).toBe('1px solid var(--ui-stroke-tertiary)')
    expect(declaration('border-radius')).toMatch(/^var\(--radius-/)
    expect(declaration('backdrop-filter')).toMatch(/^blur\([\d.]+rem\)/)
    expect(declaration('box-shadow')).toContain('inset 0 1px 0')
    expect(declaration('padding')).not.toBe('')
  })

  it('keeps the pane at 90%+ coverage — glass you can read through is not readable', () => {
    expect(coverage('color-mix(in srgb, var(--ui-bg-chrome) 86%, transparent)')).toBeCloseTo(0.86, 4)

    const fill = declaration('background')

    expect(coverage(fill)).toBeGreaterThanOrEqual(0.9)
    expect(fill).toMatch(/var\(--ui-bg-chrome\)\s+(?:9\d|100)%/)
    // Utilities paint later in the cascade (`bg-(--ui-chat-surface-background)`
    // resolves to transparent under [data-hermes-glass]); the pane's own fill
    // has to win anyway.
    expect(fill).toContain('!important')
  })

  it('paints the pane with full ink, never raw text on the material', () => {
    expect(declaration('color')).toBe('var(--ui-text-primary)')
    expect(paneRule?.body).toMatch(/font-weight:\s*400/)
  })

  it('keeps the preview at conversation scale, not display type', () => {
    const scale = paneRule?.body ?? ''

    // The reported surface was `text-xl leading-7` (20px/28px). 13px is the
    // app default; the pane reads at the transcript pane's 12px.
    expect(scale).toMatch(/--conversation-text-font-size:\s*[\d.]+rem/)
    expect(remOf(scale, '--conversation-text-font-size')).toBeLessThanOrEqual(0.75)
    expect(declaration('font-size')).toBe('var(--conversation-text-font-size)')
    expect(remOf(scale, '--conversation-line-height')).toBeLessThanOrEqual(1)
    expect(Number(declaration('line-height'))).toBeLessThanOrEqual(1.5)

    // ...and the markdown leaves inherit that scale instead of re-inflating.
    const heading = paneChildren.find(rule => /:is\(h1, h2, h3, h4, h5, h6\)/.test(rule.selector))
    const cell = paneChildren.find(rule => /:is\(th, td\)/.test(rule.selector))

    expect(heading).toBeDefined()
    expect(remOf(heading!.body)).toBeLessThanOrEqual(0.8125)
    expect(remOf(heading!.body)).toBeLessThan(1.25)
    const headingLeading = Number(heading!.body.match(/line-height:\s*([\d.]+)/)?.[1] ?? '99')
    expect(headingLeading).toBeLessThanOrEqual(1.3)
    expect(cell).toBeDefined()
    expect(remOf(cell!.body)).toBeLessThanOrEqual(0.6875)
  })

  it('caps the height so the orb keeps the surface', () => {
    expect(declaration('max-height')).toBe('40vh')
    expect(declaration('overflow')).toBe('hidden auto')
    expect(declaration('overscroll-behavior')).toBe('contain')
    expect(declaration('max-width')).toBe('100%')
  })

  it('lets nothing escape sideways — fences and tables scroll, prose and links wrap', () => {
    const fence = paneChildren.find(rule => /:is\(pre, \.aui-shiki, \.aui-shiki > pre\)/.test(rule.selector))
    const table = paneChildren.find(rule => /:is\(\.aui-md-table, table\)/.test(rule.selector))
    const tableScroll = paneChildren.find(rule => /\.aui-md-table$/.test(rule.selector))
    const link = paneChildren.find(rule => /\.aui-md a\b/.test(rule.selector))
    const words = paneChildren.find(rule => /overflow-wrap: break-word/.test(rule.body))

    expect(table?.body).toMatch(/max-width:\s*100%/)
    expect(tableScroll?.body).toMatch(/overflow-x:\s*auto/)
    expect(fence?.body).toMatch(/max-width:\s*100%/)
    expect(fence?.body).toMatch(/overflow-x:\s*auto/)
    expect(link?.body).toMatch(/overflow-wrap:\s*anywhere/)
    expect(words).toBeDefined()
    // Nothing may pin the preview open sideways.
    expect(paneRule?.body ?? '').not.toMatch(/white-space:\s*(?:nowrap|pre)/)
  })

  it('clears WCAG AA against its own fill in both themes', () => {
    expect(contrast(hexToRgb('#000000'), hexToRgb('#ffffff'))).toBeCloseTo(21, 1)

    for (const mode of ['light', 'dark'] as const) {
      const { chromeTint, ink, mixChrome, seed } = fillInputs(mode)
      const chrome = mix(seed, mixChrome, chromeTint)
      // `--ui-text-primary` is `--ui-base` (the theme foreground) at 94%; the
      // fill is that tint at 5% inside the chrome, then 96% over the wallpaper
      // (the residual 4% is the wall itself, the glass is meant to show it).
      const fill = mix(chrome, 0.95, over(ink, 0.94, chrome))
      const painted = over(ink, 0.94, fill)

      expect(contrast(painted, fill)).toBeGreaterThanOrEqual(4.5)
    }
  })

  it('keeps the transcript pane contract untouched', () => {
    const frameRules = rules.filter(rule => rule.selector.includes('[data-chat-transcript-frame]'))

    expect(frameRules.some(rule => rule.selector.includes(':root[data-hermes-glass]'))).toBe(true)
    expect(frameRules.some(rule => rule.selector.includes(':root[data-hermes-clear]'))).toBe(true)
  })
})
