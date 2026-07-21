import { describe, expect, it } from 'vitest'

import { BUILTIN_THEME_LIST, DEFAULT_TYPOGRAPHY, EMOJI_FALLBACK } from './presets'

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

describe('DesktopThemeTypography optional sizing fields (#41766)', () => {
  it('DEFAULT_TYPOGRAPHY does not set baseSize/lineHeight/letterSpacing (uses CSS defaults)', () => {
    expect(DEFAULT_TYPOGRAPHY).not.toHaveProperty('baseSize')
    expect(DEFAULT_TYPOGRAPHY).not.toHaveProperty('lineHeight')
    expect(DEFAULT_TYPOGRAPHY).not.toHaveProperty('letterSpacing')
  })

  it('theme presets with typography overrides may include sizing fields', () => {
    // All built-in themes should still be valid even if they don't set these fields.
    for (const theme of BUILTIN_THEME_LIST) {
      const typo = theme.typography
      if (typo?.baseSize !== undefined) {
        expect(typeof typo.baseSize).toBe('string')
      }
      if (typo?.lineHeight !== undefined) {
        expect(typeof typo.lineHeight).toBe('string')
      }
      if (typo?.letterSpacing !== undefined) {
        expect(typeof typo.letterSpacing).toBe('string')
      }
    }
  })
})
