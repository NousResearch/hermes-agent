import { describe, expect, it } from 'vitest'

import type { DesktopMarketplaceThemeResult } from '@/global'

import { luminance } from './color'
import { buildThemeFromMarketplace } from './install'

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

  // Same-mode palettes can't fold into a colors/darkColors pair.
  it('keeps same-mode flavors as variants, keyed by mode and short label', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'metaphore.kanagawa-vscode-color-theme',
      displayName: 'Kanagawa Flavors',
      themes: [
        { label: 'Kanagawa Wave', uiTheme: 'vs-dark', contents: themeJson('dark', '#1f1f28', '#dcd7ba') },
        { label: 'Kanagawa Dragon', uiTheme: 'vs-dark', contents: themeJson('dark', '#181616', '#c5c9c5') },
        { label: 'Kanagawa Lotus', uiTheme: 'vs-dark', contents: themeJson('dark', '#f2ecbc', '#545464') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)

    // The shared "Kanagawa" prefix is dropped for a compact picker.
    expect(theme.variants?.map(variant => variant.label)).toEqual(['Wave', 'Dragon', 'Lotus'])
    expect(theme.variants?.map(variant => variant.name)).toEqual(['wave', 'dragon', 'lotus'])
    expect(theme.variants?.every(variant => variant.mode === 'dark')).toBe(true)
    expect(theme.variants?.[1].colors.background).toBe('#181616')
    // The family pair still resolves for code paths that don't know variants.
    expect(theme.darkColors?.background).toBe('#1f1f28')
  })

  it('keeps a plain light + dark pair variant-free', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'github.github-vscode-theme',
      displayName: 'GitHub Theme',
      themes: [
        { label: 'GitHub Dark Default', uiTheme: 'vs-dark', contents: themeJson('dark', '#0d1117', '#e6edf3') },
        { label: 'GitHub Light Default', uiTheme: 'vs', contents: themeJson('light', '#ffffff', '#1f2328') }
      ]
    }

    expect(buildThemeFromMarketplace(result).variants).toBeUndefined()
  })

  it('offers each flavor of a mixed family in contribution order', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'catppuccin.catppuccin-vsc',
      displayName: 'Catppuccin',
      themes: [
        { label: 'Catppuccin Latte', uiTheme: 'vs', contents: themeJson('light', '#eff1f5', '#4c4f69') },
        { label: 'Catppuccin Frappe', uiTheme: 'vs-dark', contents: themeJson('dark', '#303446', '#c6d0f5') },
        { label: 'Catppuccin Macchiato', uiTheme: 'vs-dark', contents: themeJson('dark', '#24273a', '#cad3f5') },
        { label: 'Catppuccin Mocha', uiTheme: 'vs-dark', contents: themeJson('dark', '#1e1e2e', '#cdd6f4') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)

    expect(theme.colors.background).toBe('#eff1f5')
    expect(theme.darkColors?.background).toBe('#303446')
    expect(theme.variants?.map(variant => variant.label)).toEqual(['Latte', 'Frappe', 'Macchiato', 'Mocha'])
  })

  it('never strips a prefix that would empty a label', () => {
    const result: DesktopMarketplaceThemeResult = {
      extensionId: 'dracula-theme.theme-dracula',
      displayName: 'Dracula',
      themes: [
        { label: 'Dracula', uiTheme: 'vs-dark', contents: themeJson('dark', '#282a36', '#f8f8f2') },
        { label: 'Dracula Soft', uiTheme: 'vs-dark', contents: themeJson('dark', '#22222c', '#f8f8f2') }
      ]
    }

    const theme = buildThemeFromMarketplace(result)

    expect(theme.variants?.map(variant => variant.label)).toEqual(['Dracula', 'Dracula Soft'])
    expect(theme.variants?.map(variant => variant.name)).toEqual(['dracula', 'dracula-soft'])
  })
})
