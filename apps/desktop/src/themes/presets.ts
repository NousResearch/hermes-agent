/**
 * Built-in desktop themes. Names match the CLI skins / dashboard presets.
 * Add new themes here — no code changes needed elsewhere.
 */

import { THEME_PRESET_PALETTES } from '@hermes/shared'

import type { DesktopTheme, DesktopThemeTypography } from './types'

// Color-emoji fonts to append to every stack as a last resort. None of the UI
// text/mono fonts carry emoji glyphs, so without this emoji render as tofu
// boxes on platforms whose default text font lacks them (e.g. Linux/#40364).
// Covers macOS, Windows, Linux, plus the `emoji` generic for anything else.
export const EMOJI_FALLBACK = '"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", emoji'

const SYSTEM_SANS =
  '"Segoe WPC", "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", system-ui, sans-serif, ' +
  EMOJI_FALLBACK

// Keep the bundled JetBrains Mono first in every mono fallback chain. Several
// theme display fonts (Courier Prime, IBM Plex Mono) do not cover Vietnamese /
// Latin Extended glyphs; falling directly to platform UI fonts makes code
// blocks visibly non-monospace. JetBrains Mono is bundled with the app and
// covers those glyphs, while the Linux fallbacks catch distros without it.
const SYSTEM_MONO =
  '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", ' +
  '"Noto Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace, ' +
  EMOJI_FALLBACK

export const DEFAULT_TYPOGRAPHY: DesktopThemeTypography = { fontSans: SYSTEM_SANS, fontMono: SYSTEM_MONO }

// ---------------------------------------------------------------------------
// Brand-inspired built-in themes
// ---------------------------------------------------------------------------

export const microsoftTheme: DesktopTheme = {
  name: 'microsoft',
  label: 'Microsoft',
  description: 'Clean Microsoft 365 blue on white',
  ...THEME_PRESET_PALETTES.microsoft,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const appleTheme: DesktopTheme = {
  name: 'apple',
  label: 'Apple',
  description: 'Soft silver and San Francisco blue',
  ...THEME_PRESET_PALETTES.apple,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const googleTheme: DesktopTheme = {
  name: 'google',
  label: 'Google',
  description: 'Material Design with Google Blue',
  ...THEME_PRESET_PALETTES.google,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const chatgptTheme: DesktopTheme = {
  name: 'chatgpt',
  label: 'ChatGPT',
  description: 'OpenAI green on clean chat surfaces',
  ...THEME_PRESET_PALETTES.chatgpt,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const claudeTheme: DesktopTheme = {
  name: 'claude',
  label: 'Claude',
  description: 'Warm beige with Anthropic orange',
  ...THEME_PRESET_PALETTES.claude,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const grokTheme: DesktopTheme = {
  name: 'grok',
  label: 'Grok',
  description: 'Stark black and white — xAI minimalism',
  ...THEME_PRESET_PALETTES.grok,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const githubTheme: DesktopTheme = {
  name: 'github',
  label: 'GitHub',
  description: 'GitHub Light Default and Dark Default',
  ...THEME_PRESET_PALETTES.github,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const midnightTheme: DesktopTheme = {
  name: 'midnight',
  label: 'Midnight',
  description: 'Deep blue-violet with cool accents',
  ...THEME_PRESET_PALETTES.midnight,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const slateTheme: DesktopTheme = {
  name: 'slate',
  label: 'Slate',
  description: 'Cool slate blue — focused and professional',
  ...THEME_PRESET_PALETTES.slate,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const forestTheme: DesktopTheme = {
  name: 'forest',
  label: 'Forest',
  description: 'Warm, low-contrast forest greens',
  ...THEME_PRESET_PALETTES.forest,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const unitalkTheme: DesktopTheme = {
  name: 'unitalk',
  label: 'Unitalk',
  description: 'Warm paper & magenta — the Unitalk brand',
  ...THEME_PRESET_PALETTES.unitalk,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  }
}

export const BUILTIN_THEMES: Record<string, DesktopTheme> = {
  microsoft: microsoftTheme,
  apple: appleTheme,
  google: googleTheme,
  chatgpt: chatgptTheme,
  claude: claudeTheme,
  grok: grokTheme,
  github: githubTheme,
  midnight: midnightTheme,
  slate: slateTheme,
  forest: forestTheme,
  unitalk: unitalkTheme
}

export const BUILTIN_THEME_LIST = Object.values(BUILTIN_THEMES)

/** Skin used when nothing is persisted or the persisted name is retired. */
export const DEFAULT_SKIN_NAME = 'unitalk'

/** Names that no longer resolve to a skin of their own. A stored pick of one
 *  falls back to DEFAULT_SKIN_NAME, and a cached backend theme under one is
 *  dropped. */
export const RETIRED_SKINS = new Set<string>()