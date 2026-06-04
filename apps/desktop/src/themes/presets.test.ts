import { describe, expect, it } from 'vitest'

import { BUILTIN_THEME_LIST, BUILTIN_THEMES, DEFAULT_TYPOGRAPHY, EMOJI_FALLBACK, hepburnTheme, nousTheme } from './presets'

// #40364: none of the UI text/mono fonts carry emoji glyphs, so every font
// stack must end with a color-emoji fallback or emoji render as tofu on
// platforms whose default font lacks them (e.g. Linux).
describe('theme typography emoji fallback (#40364)', () => {
  const stacks: Array<[string, string]> = [
    ['DEFAULT_TYPOGRAPHY.fontSans', DEFAULT_TYPOGRAPHY.fontSans],
    ['DEFAULT_TYPOGRAPHY.fontMono', DEFAULT_TYPOGRAPHY.fontMono],
    // A theme may override only fontMono (fontSans then falls back to the
    // default, which already carries the emoji stack), so skip undefined.
    ...BUILTIN_THEME_LIST.flatMap(theme =>
      (
        [
          [`${theme.name}.fontSans`, theme.typography?.fontSans],
          [`${theme.name}.fontMono`, theme.typography?.fontMono]
        ] as Array<[string, string | undefined]>
      ).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
    )
  ]

  it.each(stacks)('%s includes a color-emoji font', (_label, stack) => {
    expect(stack).toMatch(/Apple Color Emoji|Segoe UI Emoji|Noto Color Emoji|(^|,\s*)emoji\b/)
  })

  it('EMOJI_FALLBACK lists the major platform emoji fonts', () => {
    expect(EMOJI_FALLBACK).toContain('Apple Color Emoji')
    expect(EMOJI_FALLBACK).toContain('Segoe UI Emoji')
    expect(EMOJI_FALLBACK).toContain('Noto Color Emoji')
  })
})

describe('desktop theme presets', () => {
  it('ports Hepburn colors from the Hermes WebUI skin', () => {
    expect(BUILTIN_THEMES.hepburn).toBe(hepburnTheme)
    expect(hepburnTheme.description).toContain('Hermes WebUI')

    expect(hepburnTheme.colors).toMatchObject({
      background: '#fff3f7',
      foreground: '#3d1a28',
      primary: '#d44a7a',
      sidebarBackground: '#fbe4ed'
    })

    expect(hepburnTheme.darkColors).toMatchObject({
      background: '#110a0f',
      foreground: '#f2e4ee',
      card: '#241420',
      primary: '#f278ad',
      sidebarBackground: '#1e0f19'
    })

    expect(hepburnTheme.typography).toEqual(nousTheme.typography)
  })
})
