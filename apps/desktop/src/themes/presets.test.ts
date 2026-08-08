import { describe, expect, it } from 'vitest'

import { BUILTIN_THEME_LIST, DEFAULT_TYPOGRAPHY, EMOJI_FALLBACK, NERD_GLYPH_FALLBACK, withNerdGlyphFallback } from './presets'

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

  it('keeps Nerd Font icon fallbacks available to the UI', () => {
    expect(DEFAULT_TYPOGRAPHY.fontSans).toContain(NERD_GLYPH_FALLBACK)
    expect(DEFAULT_TYPOGRAPHY.fontMono).toContain(NERD_GLYPH_FALLBACK)
  })

  it('adds the private-use fallback to external theme typography without changing its preferred stack', () => {
    expect(withNerdGlyphFallback('ui-monospace, monospace')).toBe(
      `${NERD_GLYPH_FALLBACK}, ui-monospace, monospace`
    )
  })

  it('moves and deduplicates the private-use fallback ahead of external theme fonts', () => {
    expect(withNerdGlyphFallback(`ui-monospace, ${NERD_GLYPH_FALLBACK}, monospace`)).toBe(
      `${NERD_GLYPH_FALLBACK}, ui-monospace, monospace`
    )
    expect(withNerdGlyphFallback('')).toBe(NERD_GLYPH_FALLBACK)
  })
})
