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
  JSON.stringify({ type, colors: { 'editor.background': background, 'editor.foreground': foreground, ...ansiColors(red) } })

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
        { label: 'Solarized Light', uiTheme: 'vs', contents: themeJsonWithAnsi('light', '#fdf6e3', '#586e75', '#dc322f') },
        { label: 'Solarized Dark', uiTheme: 'vs-dark', contents: themeJsonWithAnsi('dark', '#002b36', '#93a1a1', '#ff5f56') }
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
      themes: [{ label: 'Dracula', uiTheme: 'vs-dark', contents: themeJsonWithAnsi('dark', '#282a36', '#f8f8f2', '#ff5555') }]
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
    expect(() =>
      buildThemeFromMarketplace({ extensionId: 'x.y', displayName: 'X', themes: [] })
    ).toThrow(/does not contribute/i)
  })
})
