import { describe, expect, it } from 'vitest'

import type { DesktopMarketplaceThemeResult } from '@/global'

import { luminance } from './color'
import { buildThemeFromMarketplace } from './install'

const themeJson = (type: 'light' | 'dark', background: string, foreground: string) =>
  JSON.stringify({ type, colors: { 'editor.background': background, 'editor.foreground': foreground } })

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

  it('throws when the extension contributes no themes', () => {
    expect(() =>
      buildThemeFromMarketplace({ extensionId: 'x.y', displayName: 'X', themes: [] })
    ).toThrow(/does not contribute/i)
  })
})
