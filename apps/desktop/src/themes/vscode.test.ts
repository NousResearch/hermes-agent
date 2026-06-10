import { describe, expect, it } from 'vitest'

import { contrastRatio } from './color'
import { convertVscodeColorTheme, parseVscodeTheme, vscodeThemeSlug } from './vscode'

describe('vscodeThemeSlug', () => {
  it('namespaces, lowercases, and dashes', () => {
    expect(vscodeThemeSlug('Dracula Soft')).toBe('vsc-dracula-soft')
    expect(vscodeThemeSlug('  One Dark Pro!! ')).toBe('vsc-one-dark-pro')
  })

  it('falls back when the name has no usable characters', () => {
    expect(vscodeThemeSlug('—')).toBe('vsc-theme')
  })
})

describe('parseVscodeTheme (JSONC tolerance)', () => {
  it('strips comments and trailing commas', () => {
    const text = `{
      // a line comment
      "name": "Demo",
      /* block comment */
      "type": "dark",
      "colors": {
        "editor.background": "#1e1e2e", // inline
      },
    }`

    const parsed = parseVscodeTheme(text)
    expect(parsed.name).toBe('Demo')
    expect(parsed.colors?.['editor.background']).toBe('#1e1e2e')
  })

  it('throws on a non-object', () => {
    expect(() => parseVscodeTheme('42')).toThrow()
  })
})

describe('convertVscodeColorTheme', () => {
  const dracula = {
    name: 'Dracula',
    type: 'dark',
    colors: {
      'editor.background': '#282a36',
      'editor.foreground': '#f8f8f2',
      focusBorder: '#6272a4',
      'editorWidget.background': '#21222c',
      'sideBar.background': '#21222c',
      errorForeground: '#ff5555',
      // 8-digit hex (alpha) — must flatten over the background.
      'panel.border': '#bd93f900'
    }
  }

  it('maps the load-bearing tokens onto the palette', () => {
    const { theme } = convertVscodeColorTheme(dracula, { source: 'dracula-theme.theme-dracula' })

    expect(theme.name).toBe('vsc-dracula')
    expect(theme.label).toBe('Dracula')
    expect(theme.description).toContain('dracula-theme.theme-dracula')
    expect(theme.colors.background).toBe('#282a36')
    expect(theme.colors.foreground).toBe('#f8f8f2')
    // One accent drives primary + ring + midground together...
    expect(theme.colors.ring).toBe(theme.colors.primary)
    expect(theme.colors.midground).toBe(theme.colors.primary)
    // ...and it's nudged until it reads on the sidebar it labels (the dim
    // focusBorder #6272a4 sits below AA, so it's lifted).
    expect(contrastRatio(theme.colors.primary, theme.colors.sidebarBackground!)).toBeGreaterThanOrEqual(4.5)
    expect(theme.colors.popover).toBe('#21222c')
    expect(theme.colors.sidebarBackground).toBe('#21222c')
    expect(theme.colors.destructive).toBe('#ff5555')
  })

  it('flattens alpha hex over the background (no #rrggbbaa leaks)', () => {
    const { theme } = convertVscodeColorTheme(dracula)
    expect(theme.colors.border).toMatch(/^#[0-9a-f]{6}$/)
    // 00 alpha over the bg means the border collapses to the background.
    expect(theme.colors.border).toBe('#282a36')
  })

  it('renders identically in both modes (single palette in both slots)', () => {
    const { theme } = convertVscodeColorTheme(dracula)
    expect(theme.darkColors).toBe(theme.colors)
  })

  it('records derived fallbacks for omitted tokens', () => {
    const { derived } = convertVscodeColorTheme({
      name: 'Sparse',
      type: 'dark',
      colors: { 'editor.background': '#101010', 'editor.foreground': '#fafafa' }
    })

    // No accent/elevated/sidebar/error tokens → all derived. The accent records
    // its first candidate (button.background) when none of the family is present.
    expect(derived).toContain('button.background')
    expect(derived).toContain('editorWidget.background')
    expect(derived).toContain('editorError.foreground')
  })

  it('buckets light vs dark from background luminance when type is absent', () => {
    const light = convertVscodeColorTheme({
      name: 'Bright',
      colors: { 'editor.background': '#ffffff', 'editor.foreground': '#1a1a1a' }
    }).theme

    // A light background should keep a near-white background, not synth dark.
    expect(light.colors.background).toBe('#ffffff')
  })

  it('throws when there is no colors map', () => {
    expect(() => convertVscodeColorTheme({ name: 'Empty' })).toThrow(/colors/)
  })
})
