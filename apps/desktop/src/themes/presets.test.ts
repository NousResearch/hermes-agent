import { describe, expect, it } from 'vitest'

import { BUILTIN_THEME_LIST, DEFAULT_TYPOGRAPHY, EMOJI_FALLBACK, hadesTheme } from './presets'

const DESKTOP_COLOR_KEYS = [
  'background',
  'foreground',
  'card',
  'cardForeground',
  'muted',
  'mutedForeground',
  'popover',
  'popoverForeground',
  'primary',
  'primaryForeground',
  'secondary',
  'secondaryForeground',
  'accent',
  'accentForeground',
  'border',
  'input',
  'ring',
  'midground',
  'midgroundForeground',
  'composerRing',
  'destructive',
  'destructiveForeground',
  'sidebarBackground',
  'sidebarBorder',
  'userBubble',
  'userBubbleBorder'
] as const

const TERMINAL_COLOR_KEYS = [
  'foreground',
  'cursor',
  'selectionBackground',
  'black',
  'red',
  'green',
  'yellow',
  'blue',
  'magenta',
  'cyan',
  'white',
  'brightBlack',
  'brightRed',
  'brightGreen',
  'brightYellow',
  'brightBlue',
  'brightMagenta',
  'brightCyan',
  'brightWhite'
] as const

describe('Hades desktop theme completeness', () => {
  it('sets every base and dark desktop color explicitly', () => {
    for (const key of DESKTOP_COLOR_KEYS) {
      expect(hadesTheme.colors[key], `hades.colors.${key}`).toMatch(/^#[0-9A-F]{6}$/i)
      expect(hadesTheme.darkColors?.[key], `hades.darkColors.${key}`).toBe(hadesTheme.colors[key])
    }
  })

  it('sets every integrated-terminal ANSI color for light and dark terminal modes', () => {
    for (const key of TERMINAL_COLOR_KEYS) {
      expect(hadesTheme.terminal?.[key], `hades.terminal.${key}`).toMatch(/^#[0-9A-F]{6}$/i)
      expect(hadesTheme.darkTerminal?.[key], `hades.darkTerminal.${key}`).toBe(hadesTheme.terminal?.[key])
    }
  })
})

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
