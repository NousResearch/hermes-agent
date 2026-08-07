import { beforeEach, describe, expect, it } from 'vitest'

import { contrastRatio } from './color'
import {
  $customThemeDefinitions,
  $themePreview,
  buildCustomTheme,
  createCustomThemeDefinition,
  type CustomThemePaletteSeed,
  DEFAULT_CUSTOM_DARK_PALETTE,
  DEFAULT_CUSTOM_LIGHT_PALETTE,
  deriveCustomThemeColors,
  getCustomThemeDefinition,
  isCustomTheme,
  removeCustomTheme,
  resetCustomThemePalettes,
  saveCustomTheme,
  uniqueCustomThemeName
} from './custom-themes'
import { nousTheme } from './presets'
import { $userThemes, installUserTheme, isUserTheme, resolveTheme } from './user-themes'

const CUSTOM_THEMES_KEY = 'hermes-desktop-custom-theme-definitions-v1'

const seed = (contrast = 38): CustomThemePaletteSeed => ({
  accent: '#cc7d5e',
  background: '#f9f9f7',
  foreground: '#2d2d2b',
  contrast
})

const definition = () =>
  createCustomThemeDefinition({
    source: nousTheme,
    lightColors: nousTheme.colors,
    darkColors: nousTheme.darkColors ?? nousTheme.colors,
    label: 'Warm Paper'
  })

describe('custom theme derivation', () => {
  it('guarantees readable body text and a visible key accent', () => {
    const colors = deriveCustomThemeColors(
      {
        accent: '#fefefe',
        background: '#ffffff',
        foreground: '#fdfdfd',
        contrast: 38
      },
      false
    )

    expect(contrastRatio(colors.foreground, colors.background)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(colors.primary, colors.background)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(colors.primary, colors.sidebarBackground ?? colors.background)).toBeGreaterThanOrEqual(3)
  })

  it('uses contrast only for surface and border depth', () => {
    const flat = deriveCustomThemeColors(seed(0), false)
    const strong = deriveCustomThemeColors(seed(100), false)

    expect(strong.background).toBe(flat.background)
    expect(strong.foreground).toBe(flat.foreground)
    expect(strong.primary).toBe(flat.primary)
    expect(strong.card).not.toBe(flat.card)
    expect(strong.border).not.toBe(flat.border)
  })

  it('keeps translucency scoped to the sidebar surface', () => {
    const colors = deriveCustomThemeColors(seed(), true)

    expect(colors.background).toBe('#f9f9f7')
    expect(colors.card).toMatch(/^#/)
    expect(colors.sidebarBackground).toContain('82%')
    expect(colors.sidebarBackground).toContain('transparent')
  })
})

describe('custom theme lifecycle', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $customThemeDefinitions.set({})
    $userThemes.set({})
    $themePreview.set(null)
  })

  it('creates unique stable slugs without shadowing an installed theme', () => {
    const first = definition()
    saveCustomTheme(first)

    expect(uniqueCustomThemeName('Warm Paper')).toBe('custom-warm-paper-2')
    expect(uniqueCustomThemeName('Warm Paper', first.name)).toBe(first.name)
  })

  it('copies both palettes from the active built-in theme', () => {
    const draft = definition()

    expect(draft.light).toEqual({
      accent: nousTheme.colors.midground!.toLowerCase(),
      background: nousTheme.colors.background.toLowerCase(),
      foreground: nousTheme.colors.foreground.toLowerCase(),
      contrast: 38
    })
    expect(draft.dark).toEqual({
      accent: (nousTheme.darkColors ?? nousTheme.colors).midground!.toLowerCase(),
      background: (nousTheme.darkColors ?? nousTheme.colors).background.toLowerCase(),
      foreground: (nousTheme.darkColors ?? nousTheme.colors).foreground.toLowerCase(),
      contrast: 38
    })
  })

  it('copies palettes from a backend theme that is not installed as a user theme', () => {
    const source = {
      ...nousTheme,
      name: 'backend-paper',
      label: 'Backend Paper'
    }

    const lightColors = {
      ...source.colors,
      background: '#fff4dc',
      foreground: '#2b2118',
      midground: '#a34f31'
    }

    const darkColors = {
      ...(source.darkColors ?? source.colors),
      background: '#17120f',
      foreground: '#f7eee7',
      midground: '#db8666'
    }

    const draft = createCustomThemeDefinition({ source, lightColors, darkColors })

    expect(draft.light).toEqual({
      accent: '#a34f31',
      background: '#fff4dc',
      foreground: '#2b2118',
      contrast: 38
    })
    expect(draft.dark).toEqual({
      accent: '#db8666',
      background: '#17120f',
      foreground: '#f7eee7',
      contrast: 38
    })
  })

  it('resets both palettes without discarding theme identity or appearance options', () => {
    const draft = definition()
    draft.label = 'My Working Theme'
    draft.light = { accent: '#112233', background: '#ffffff', foreground: '#111111', contrast: 12 }
    draft.dark = { accent: '#445566', background: '#111111', foreground: '#ffffff', contrast: 91 }
    draft.fontSans = 'Charter, Georgia, serif'
    draft.fontMono = '"SF Mono", monospace'
    draft.translucentSidebar = true

    const reset = resetCustomThemePalettes(draft)

    expect(reset.light).toEqual(DEFAULT_CUSTOM_LIGHT_PALETTE)
    expect(reset.dark).toEqual(DEFAULT_CUSTOM_DARK_PALETTE)
    expect(reset).toMatchObject({
      name: draft.name,
      label: 'My Working Theme',
      fontSans: 'Charter, Georgia, serif',
      fontMono: '"SF Mono", monospace',
      translucentSidebar: true
    })
    expect(draft.light.accent).toBe('#112233')
    expect(draft.dark.accent).toBe('#445566')
  })

  it('saves editable seeds and a standard DesktopTheme for boot-time reload', () => {
    const draft = definition()
    draft.light.background = '#fffaf2'
    draft.fontSans = 'Charter, Georgia, serif'
    const theme = saveCustomTheme(draft)

    const stored = JSON.parse(window.localStorage.getItem(CUSTOM_THEMES_KEY) ?? '{}') as {
      version?: number
      themes?: Record<string, unknown>
    }

    expect(stored.version).toBe(1)
    expect(stored.themes?.[draft.name]).toBeTruthy()
    expect(getCustomThemeDefinition(draft.name)?.light.background).toBe('#fffaf2')
    expect(resolveTheme(draft.name)).toEqual(theme)
    expect(window.localStorage.getItem('hermes-desktop-user-themes-v1')).toContain(draft.name)
  })

  it('deletes both generated and editable theme data', () => {
    const draft = definition()
    saveCustomTheme(draft)
    removeCustomTheme(draft.name)

    expect(isCustomTheme(draft.name)).toBe(false)
    expect(isUserTheme(draft.name)).toBe(false)
    expect(window.localStorage.getItem(CUSTOM_THEMES_KEY)).not.toContain(draft.name)
  })

  it('keeps live preview ephemeral until save', () => {
    const draft = definition()

    $themePreview.set({ mode: 'light', theme: buildCustomTheme(draft) })

    expect(window.localStorage.getItem(CUSTOM_THEMES_KEY)).toBeNull()
    expect(window.localStorage.getItem('hermes-desktop-user-themes-v1')).toBeNull()

    $themePreview.set(null)
    expect($themePreview.get()).toBeNull()
  })

  it('copies an imported theme without mutating the source', () => {
    const source = {
      ...nousTheme,
      name: 'vsc-paper',
      label: 'Paper',
      colors: { ...nousTheme.colors }
    }

    const before = structuredClone(source)
    installUserTheme(source)

    const draft = createCustomThemeDefinition({
      source,
      lightColors: source.colors,
      darkColors: source.darkColors ?? source.colors
    })

    draft.light.accent = '#a14f32'
    saveCustomTheme(draft)

    expect(source).toEqual(before)
    expect(draft.name).not.toBe(source.name)
    expect(draft.light.background).toBe(source.colors.background.toLowerCase())
    expect(draft.dark.background).toBe(source.darkColors?.background.toLowerCase())
  })
})
