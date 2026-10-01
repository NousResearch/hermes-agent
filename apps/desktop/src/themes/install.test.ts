import { describe, expect, it, vi } from 'vitest'

import type { DesktopMarketplaceThemeResult } from '@/global'

import { luminance } from './color'
import { getBaseColors } from './context'
import { buildThemeFromMarketplace, installVscodeThemeFromMarketplace } from './install'
import { $marketplaceInstalls, $userThemes, removeUserTheme } from './user-themes'

const themeJson = (type: 'light' | 'dark', background: string, foreground: string) =>
  JSON.stringify({ type, colors: { 'editor.background': background, 'editor.foreground': foreground } })

// A full base-8 ANSI set keyed off `red` so each variant is distinguishable.
const ansiColors = (red: string) => ({
  'terminal.ansiBlack': '#000000',
  'terminal.ansiRed': red,
  'terminal.ansiGreen': '#00aa00',
  'terminal.ansiYellow': '#aaaa00',
  'terminal.ansiBlue': '#0000aa',
  'terminal.ansiMagenta': '#aa00aa',
  'terminal.ansiCyan': '#00aaaa',
  'terminal.ansiWhite': '#aaaaaa'
})

const themeJsonWithAnsi = (type: 'light' | 'dark', background: string, foreground: string, red: string) =>
  JSON.stringify({
    type,
    colors: { 'editor.background': background, 'editor.foreground': foreground, ...ansiColors(red) }
  })

describe('buildThemeFromMarketplace', () => {
  it('folds a light + dark variant into one family with both slots', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'ryanolsonx.solarized',
      displayName: 'Solarized',
      themes: [
        { label: 'Solarized Light', uiTheme: 'vs', contents: themeJson('light', '#fdf6e3', '#586e75') },
        { label: 'Solarized Dark', uiTheme: 'vs-dark', contents: themeJson('dark', '#002b36', '#93a1a1') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)

    expect(theme.label).toBe('Solarized')
    expect(theme.name).toBe('vsc-solarized')
    // colors = the light variant, darkColors = the dark variant → the toggle works.
    expect(theme.colors.background).toBe('#fdf6e3')
    expect(theme.darkColors?.background).toBe('#002b36')
    expect(luminance(theme.colors.background)).toBeGreaterThan(0.5)
    expect(luminance(theme.darkColors!.background)).toBeLessThan(0.5)
  })

  it('orders variants by contribution regardless of light/dark sequence', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'github.github-vscode-theme',
      displayName: 'GitHub Theme',
      themes: [
        { label: 'GitHub Dark Default', uiTheme: 'vs-dark', contents: themeJson('dark', '#0d1117', '#e6edf3') },
        { label: 'GitHub Light Default', uiTheme: 'vs', contents: themeJson('light', '#ffffff', '#1f2328') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)
    expect(theme.colors.background).toBe('#ffffff')
    expect(theme.darkColors?.background).toBe('#0d1117')
  })

  it('fills both slots with the sole palette for a single-variant extension', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'dracula-theme.theme-dracula',
      displayName: 'Dracula',
      themes: [{ label: 'Dracula', uiTheme: 'vs-dark', contents: themeJson('dark', '#282a36', '#f8f8f2') }]
    }

    const theme = buildThemeFromMarketplace(result)
    expect(theme.colors.background).toBe('#282a36')
    expect(theme.darkColors).toBe(theme.colors)
  })

  it('keys each variant terminal palette to its mode (terminal / darkTerminal)', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'ryanolsonx.solarized',
      displayName: 'Solarized',
      themes: [
        {
          label: 'Solarized Light',
          uiTheme: 'vs',
          contents: themeJsonWithAnsi('light', '#fdf6e3', '#586e75', '#dc322f')
        },
        {
          label: 'Solarized Dark',
          uiTheme: 'vs-dark',
          contents: themeJsonWithAnsi('dark', '#002b36', '#93a1a1', '#ff5f56')
        }
      ]
    }

    const theme = buildThemeFromMarketplace(result)
    expect(theme.terminal?.red).toBe('#dc322f')
    expect(theme.darkTerminal?.red).toBe('#ff5f56')
  })

  it('reuses the sole variant terminal palette for both modes', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'dracula-theme.theme-dracula',
      displayName: 'Dracula',
      themes: [
        { label: 'Dracula', uiTheme: 'vs-dark', contents: themeJsonWithAnsi('dark', '#282a36', '#f8f8f2', '#ff5555') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)
    expect(theme.terminal?.red).toBe('#ff5555')
    expect(theme.darkTerminal?.red).toBe('#ff5555')
  })

  it('leaves terminal slots unset when no variant ships an ANSI palette', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'x.plain',
      displayName: 'Plain',
      themes: [{ label: 'Plain', uiTheme: 'vs-dark', contents: themeJson('dark', '#101010', '#fafafa') }]
    }

    const theme = buildThemeFromMarketplace(result)
    expect(theme.terminal).toBeUndefined()
    expect(theme.darkTerminal).toBeUndefined()
  })

  it('throws when the extension contributes no themes', () => {
    expect(() => buildThemeFromMarketplace({ extensionId: 'x.y', displayName: 'X', themes: [] })).toThrow()
  })
})

it('installs every multi-variant palette and reselects the initial dark variant', async () => {
  const result: DesktopMarketplaceThemeResult = {
    extensionId: 'example.variants',
    displayName: 'Variants',
    themes: [
      { label: 'Light', uiTheme: 'vs', contents: themeJsonWithAnsi('light', '#ffffff', '#222222', '#cc0000') },
      { label: 'Dark', uiTheme: 'vs-dark', contents: themeJsonWithAnsi('dark', '#111111', '#eeeeee', '#dd0000') },
      { label: 'Dim', uiTheme: 'vs-dark', contents: themeJsonWithAnsi('dark', '#333333', '#dddddd', '#ee0000') }
    ]
  }

  $userThemes.set({})
  vi.stubGlobal('hermesDesktop', { themes: { fetchMarketplace: vi.fn().mockResolvedValue(result) } })

  try {
    const active = await installVscodeThemeFromMarketplace(result.extensionId)
    const stored = Object.values($userThemes.get())
    expect(stored.map(theme => theme.label).sort()).toEqual(result.themes.map(theme => theme.label).sort())
    expect(active.label).toBe('Dark')

    for (const theme of stored) {
      const source = result.themes.find(variant => variant.label === theme.label)!
      const colors = JSON.parse(source.contents).colors
      expect(theme.colors.background).toBe(colors['editor.background'])
      expect(theme.terminal?.red).toBe(colors['terminal.ansiRed'])
      expect(theme.darkColors).toBe(theme.colors)
      expect(getBaseColors(theme.name, 'light')).toBe(theme.colors)
      expect(getBaseColors(theme.name, 'dark')).toBe(theme.colors)
      expect(theme.darkTerminal).toEqual(theme.terminal)
    }

    const tracked = $marketplaceInstalls.get().get(result.extensionId)
    expect(tracked).toEqual(stored)
    expect(tracked?.[0]).toEqual(active)
  } finally {
    vi.unstubAllGlobals()
    $userThemes.set({})
    window.localStorage.clear()
  }
})

it('keeps ordinary light/dark families and makes all-light variants independently selectable', async () => {
  $userThemes.set({})

  const pair: DesktopMarketplaceThemeResult = {
    extensionId: 'example.pair',
    displayName: 'Pair',
    themes: [
      { label: 'Light', uiTheme: 'vs', contents: themeJson('light', '#ffffff', '#222222') },
      { label: 'Dark', uiTheme: 'vs-dark', contents: themeJson('dark', '#111111', '#eeeeee') }
    ]
  }

  const variants: DesktopMarketplaceThemeResult = {
    extensionId: 'example.light',
    displayName: 'Light Variants',
    themes: ['Paper', 'Cream', 'Snow'].map((label, index) => ({
      label, uiTheme: 'vs', contents: themeJson('light', ['#ffffff', '#ffffee', '#eeeeff'][index], '#222222')
    }))
  }

  const fetchMarketplace = vi.fn().mockResolvedValueOnce(pair).mockResolvedValueOnce(variants)
  vi.stubGlobal('hermesDesktop', { themes: { fetchMarketplace } })

  try {
    const family = await installVscodeThemeFromMarketplace(pair.extensionId)
    expect($marketplaceInstalls.get().get(pair.extensionId)).toEqual([family])
    expect(getBaseColors(family.name, 'light').background).toBe('#ffffff')
    expect(getBaseColors(family.name, 'dark').background).toBe('#111111')

    const active = await installVscodeThemeFromMarketplace(variants.extensionId)
    expect(active.label).toBe(variants.themes[0].label)
    expect($marketplaceInstalls.get().get(variants.extensionId)?.map(theme => theme.label))
      .toEqual(variants.themes.map(theme => theme.label))
    removeUserTheme(active.name)
    expect($marketplaceInstalls.get().get(variants.extensionId)?.map(theme => theme.label))
      .toEqual(variants.themes.slice(1).map(theme => theme.label))
  } finally {
    vi.unstubAllGlobals()
    $userThemes.set({})
    window.localStorage.clear()
  }
})
