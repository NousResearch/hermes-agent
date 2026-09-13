import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { getBaseColors, skinPref, ThemeProvider, useTheme } from './context'
import { BUILTIN_THEME_LIST, everforestTheme } from './presets'

// The live-authoring loop: Hermes writes/edits one skin file and every surface
// repaints. An in-place edit keeps the NAME — only the palette moves.
const bloomberg = (foreground: string) => ({
  name: 'bloomberg',
  colors: { background: '#000000', ui_text: foreground, ui_accent: '#ff8000' }
})

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

describe('ThemeProvider ← backend skin sync', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  afterEach(cleanup)

  it('applies an activated backend skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))

    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
    expect(cssVar('--theme-background-seed')).toBe('#000000')
  })

  it('repaints an in-place edit of the ACTIVE skin (same name, new palette)', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))
    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')

    // Recolor the same skin file. The same-name apply guard correctly no-ops
    // (protects manual desktop picks), so the repaint must come from the
    // registry update reaching the active theme derivation.
    act(() => ingestBackendSkin(bloomberg('#ff2d95'), { apply: true }))
    expect(cssVar('--theme-foreground')).toBe('#ff2d95')
  })

  it('does not repaint an edit to an INACTIVE skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))

    // A different skin registered without apply (e.g. seeded on reconnect)
    // must not touch the painted theme.
    act(() =>
      ingestBackendSkin({ name: 'forest', colors: { background: '#001100', ui_text: '#66ff66' } }, { apply: false })
    )
    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
  })

  // The relaunch bug: the persisted pick was a backend skin, and the boot paint
  // ran before the gateway seeded it. `normalizeSkin` could not resolve the
  // name, flattened it to the default, and the connect-time seed (apply: false,
  // by design) never repainted — so the theme "didn't stick" until `/skin`.
  it('paints a persisted backend skin once the connect-time seed makes it resolvable', () => {
    window.localStorage.setItem('hermes-desktop-theme-v2', 'bloomberg')

    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    // Boot: nothing resolves 'bloomberg' yet → default paint...
    expect(cssVar('--theme-background-seed')).not.toBe('#000000')

    // ...but the pick survives, so the seed alone repaints it.
    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: false }))

    expect(cssVar('--theme-background-seed')).toBe('#000000')
    expect(skinPref.resolve('default')).toBe('bloomberg')
  })
})

describe('ThemeProvider highlight preview', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  afterEach(cleanup)

  // Read the live context so the tests drive the real provider, not a mock.
  let ctx: ReturnType<typeof useTheme>

  function Probe() {
    ctx = useTheme()

    return null
  }

  const renderProbe = () =>
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )

  it('paints the previewed theme without persisting it', () => {
    renderProbe()

    const committed = ctx.themeName

    act(() => ctx.previewTheme('everforest', 'dark'))

    expect(cssVar('--theme-foreground')).toBe(everforestTheme.darkColors!.foreground)
    // The commit surface does not change. The context name and the stored
    // preference keep their values.
    expect(ctx.themeName).toBe(committed)
    expect(skinPref.resolve('default')).toBe(committed)
  })

  it('clearThemePreview repaints the committed appearance', () => {
    renderProbe()

    act(() => ctx.previewTheme('everforest', 'dark'))
    expect(cssVar('--theme-foreground')).toBe(everforestTheme.darkColors!.foreground)

    act(() => ctx.clearThemePreview())
    expect(cssVar('--theme-foreground')).not.toBe(everforestTheme.darkColors!.foreground)
  })

  it('a commit replaces the preview and persists', () => {
    renderProbe()

    act(() => ctx.previewTheme('everforest', 'dark'))
    act(() => ctx.setTheme('mono'))

    expect(ctx.themeName).toBe('mono')
    expect(skinPref.resolve('default')).toBe('mono')
    expect(cssVar('--theme-foreground')).not.toBe(everforestTheme.darkColors!.foreground)
  })

  it('ignores a preview of an unknown theme', () => {
    renderProbe()

    const painted = cssVar('--theme-foreground')

    act(() => ctx.previewTheme('does-not-exist', 'dark'))
    expect(cssVar('--theme-foreground')).toBe(painted)
  })
})

// `--dt-primary-solid` is the loud brand fill of every shipped preset. The
// desktop's original ensureContrast ladder (5 rungs of 0.2 toward the pole
// opposite the background, re-mixed from the ORIGINAL colour) is reproduced
// here as a reference implementation; the shared @hermes/shared/color ladder
// must land byte-identical for every preset in every mode, or presets change
// colour under users on an "only math moved" refactor.
describe('ThemeProvider --dt-primary-solid preset parity', () => {
  const hexToRgb = (hex: string): [number, number, number] =>
    [0, 2, 4].map(i => parseInt(hex.replace(/^#/, '').slice(i, i + 2), 16)) as [number, number, number]

  const rgbToHex = (rgb: [number, number, number]) =>
    `#${rgb.map(n => Math.round(Math.min(255, Math.max(0, n))).toString(16).padStart(2, '0')).join('')}`

  const oldMix = (a: string, b: string, amount: number) => {
    const ar = hexToRgb(a)
    const br = hexToRgb(b)

    return rgbToHex([ar[0] + (br[0] - ar[0]) * amount, ar[1] + (br[1] - ar[1]) * amount, ar[2] + (br[2] - ar[2]) * amount])
  }

  const linearize = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)

  const oldLuminance = (hex: string) => {
    const [r, g, b] = hexToRgb(hex).map(v => linearize(v / 255))

    return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!
  }

  const oldContrast = (a: string, b: string) => {
    const la = oldLuminance(a)
    const lb = oldLuminance(b)

    return la >= lb ? (la + 0.05) / (lb + 0.05) : (lb + 0.05) / (la + 0.05)
  }

  const oldEnsureContrast = (color: string, bg: string, min: number) => {
    if (oldContrast(color, bg) >= min) {
      return color
    }

    const towards = oldLuminance(bg) < 0.5 ? '#ffffff' : '#000000'
    let best = color

    for (let amount = 0.2; amount <= 1.0001; amount += 0.2) {
      best = oldMix(color, towards, Math.min(amount, 1))

      if (oldContrast(best, bg) >= min) {
        return best
      }
    }

    return best
  }

  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  afterEach(cleanup)

  let ctx: ReturnType<typeof useTheme>

  function Probe() {
    ctx = useTheme()

    return null
  }

  const cases = BUILTIN_THEME_LIST.flatMap(theme => (['light', 'dark'] as const).map(mode => [theme.name, mode] as const))

  it.each(cases)('%s/%s keeps the pre-refactor loud fill', (name, mode) => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )

    act(() => ctx.previewTheme(name, mode))

    const primary = getBaseColors(name, mode).primary
    const expected = oldEnsureContrast(primary, '#fcfcfc', 4.5)

    expect(cssVar('--dt-primary-solid')).toBe(expected)

    // At least some presets need a lift; the assertion must not be vacuous.
    if (name === 'nous' && mode === 'dark') {
      expect(expected).not.toBe(primary)
    }
  })
})
