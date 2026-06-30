import { describe, expect, it } from 'vitest'

import { algodyneCarbonTheme, BUILTIN_THEME_LIST, BUILTIN_THEMES, DEFAULT_TYPOGRAPHY, EMOJI_FALLBACK } from './presets'

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

describe('Algodyne Carbon desktop preset', () => {
  it('is registered as a built-in desktop theme', () => {
    expect(BUILTIN_THEMES['algodyne-carbon']).toBe(algodyneCarbonTheme)
    expect(BUILTIN_THEME_LIST.map(theme => theme.name)).toContain('algodyne-carbon')
  })

  it('keeps red as the brand signal over carbon surfaces', () => {
    expect(algodyneCarbonTheme.colors.background).toBe('#050505')
    expect(algodyneCarbonTheme.colors.card).toBe('#0B0B0D')
    expect(algodyneCarbonTheme.colors.primary).toBe('#FF4A4A')
    expect(algodyneCarbonTheme.colors.ring).toBe('#FF4A4A')
    expect(algodyneCarbonTheme.colors.destructive).toBe('#8F1F18')
  })

  it('reserves support colors for terminal status accents', () => {
    expect(algodyneCarbonTheme.terminal?.cyan).toBe('#5BA7E4')
    expect(algodyneCarbonTheme.terminal?.yellow).toBe('#F08A28')
    expect(algodyneCarbonTheme.terminal?.green).toBe('#2F8F71')
  })
})
