/**
 * Built-in desktop themes. Names match the CLI skins / dashboard presets.
 * Add new themes here — no code changes needed elsewhere.
 */

import type { DesktopTheme, DesktopThemeTypography } from './types'

// Color-emoji fonts to append to every stack as a last resort. None of the UI
// text/mono fonts carry emoji glyphs, so without this emoji render as tofu
// boxes on platforms whose default text font lacks them (e.g. Linux/#40364).
// Covers macOS, Windows, Linux, plus the `emoji` generic for anything else.
export const EMOJI_FALLBACK = '"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", emoji'

const SYSTEM_SANS =
  '"Segoe WPC", "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", system-ui, sans-serif, ' +
  EMOJI_FALLBACK

const SYSTEM_MONO =
  '"Cascadia Code", "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace, ' + EMOJI_FALLBACK

export const DEFAULT_TYPOGRAPHY: DesktopThemeTypography = { fontSans: SYSTEM_SANS, fontMono: SYSTEM_MONO }

const NOUS_BLUE = '#0053FD'
const PSYCHE_BLUE = '#1540B1'
const PSYCHE_WARM = '#FFE6CB'

const nousTint = (pct: number) => `color-mix(in srgb, ${NOUS_BLUE} ${pct}%, #FFFFFF)`
const nousTintTransparent = (pct: number) => `color-mix(in srgb, ${NOUS_BLUE} ${pct}%, transparent)`

/**
 * Nous — canonical Hermes desktop identity. The palette keeps the current
 * glass geometry neutral, then lets the old bb/gui blue and psyche cream
 * return as accent seeds.
 */
export const nousTheme: DesktopTheme = {
  name: 'nous',
  label: 'Nous',
  description: 'Glass neutrals with Nous blue accents',
  colors: {
    background: '#F8FAFF',
    foreground: '#17171A',
    card: '#FFFFFF',
    cardForeground: '#17171A',
    muted: nousTint(5),
    mutedForeground: '#666678',
    popover: '#FFFFFF',
    popoverForeground: '#17171A',
    primary: NOUS_BLUE,
    primaryForeground: '#FCFCFC',
    secondary: nousTint(7),
    secondaryForeground: '#242432',
    accent: nousTint(10),
    accentForeground: '#202030',
    border: nousTintTransparent(22),
    input: nousTintTransparent(30),
    ring: NOUS_BLUE,
    midground: NOUS_BLUE,
    composerRing: NOUS_BLUE,
    destructive: '#C72E4D',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: '#F3F7FF',
    sidebarBorder: nousTintTransparent(18),
    userBubble: nousTint(6),
    userBubbleBorder: nousTintTransparent(24)
  },
  darkColors: {
    background: '#0D2F86',
    foreground: PSYCHE_WARM,
    card: '#12378F',
    cardForeground: PSYCHE_WARM,
    muted: '#183F9A',
    mutedForeground: '#B5C7F3',
    popover: '#123A96',
    popoverForeground: PSYCHE_WARM,
    primary: PSYCHE_WARM,
    primaryForeground: '#0D2F86',
    secondary: '#1B45A4',
    secondaryForeground: '#E0E8FF',
    accent: PSYCHE_BLUE,
    accentForeground: '#F0F4FF',
    border: '#3158AD',
    input: '#0B2566',
    ring: PSYCHE_WARM,
    midground: NOUS_BLUE,
    composerRing: PSYCHE_WARM,
    destructive: '#C0473A',
    destructiveForeground: '#FEF2F2',
    sidebarBackground: '#09286F',
    sidebarBorder: '#234A9C',
    userBubble: '#143B91',
    userBubbleBorder: '#3A63BD'
  },
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: `"Courier Prime", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap'
  }
}

/** Premium dark glass with luminous Hermes accents. */
export const liquidGlassTheme: DesktopTheme = {
  name: 'liquid-glass',
  label: 'Liquid Glass',
  description: 'Premium dark glass surfaces with luminous Hermes accents',
  colors: {
    background: '#070914',
    foreground: '#FFE6CB',
    card: 'rgba(255, 230, 203, 0.075)',
    cardForeground: '#FFF7EF',
    muted: 'rgba(255, 230, 203, 0.08)',
    mutedForeground: '#C9B7A3',
    popover: 'rgba(13, 16, 28, 0.92)',
    popoverForeground: '#FFF7EF',
    primary: '#FFE6CB',
    primaryForeground: '#070914',
    secondary: 'rgba(255, 230, 203, 0.10)',
    secondaryForeground: '#FFE6CB',
    accent: '#00D4FF',
    accentForeground: '#061018',
    border: 'rgba(255, 230, 203, 0.16)',
    input: 'rgba(255, 230, 203, 0.13)',
    ring: '#00D4FF',
    midground: '#FFE6CB',
    composerRing: '#00D4FF',
    destructive: '#FB2C36',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: 'rgba(7, 9, 20, 0.82)',
    sidebarBorder: 'rgba(255, 230, 203, 0.12)',
    userBubble: 'rgba(255, 230, 203, 0.12)',
    userBubbleBorder: 'rgba(255, 230, 203, 0.18)'
  },
  darkColors: {
    background: '#070914',
    foreground: '#FFE6CB',
    card: 'rgba(255, 230, 203, 0.09)',
    cardForeground: '#FFF7EF',
    muted: 'rgba(255, 230, 203, 0.10)',
    mutedForeground: '#C9B7A3',
    popover: 'rgba(13, 16, 28, 0.94)',
    popoverForeground: '#FFF7EF',
    primary: '#FFE6CB',
    primaryForeground: '#070914',
    secondary: 'rgba(255, 230, 203, 0.12)',
    secondaryForeground: '#FFE6CB',
    accent: '#00D4FF',
    accentForeground: '#061018',
    border: 'rgba(255, 230, 203, 0.18)',
    input: 'rgba(255, 230, 203, 0.14)',
    ring: '#00D4FF',
    midground: '#FFE6CB',
    composerRing: '#00D4FF',
    destructive: '#FB2C36',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: 'rgba(7, 9, 20, 0.86)',
    sidebarBorder: 'rgba(255, 230, 203, 0.14)',
    userBubble: 'rgba(255, 230, 203, 0.14)',
    userBubbleBorder: 'rgba(255, 230, 203, 0.20)'
  },
  typography: {
    fontSans: '"Inter", ' + SYSTEM_SANS,
    fontMono: '"JetBrains Mono", ' + SYSTEM_MONO,
    fontUrl:
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap'
  }
}

/** Bright glass variant with Nous-blue accents and a cream canvas. */
export const liquidGlassLightTheme: DesktopTheme = {
  name: 'liquid-glass-light',
  label: 'Liquid Glass Light',
  description: 'Bright glass surfaces with Nous-blue accents and cream canvas',
  colors: {
    background: '#F7F2EA',
    foreground: '#111827',
    card: 'rgba(255, 255, 255, 0.76)',
    cardForeground: '#111827',
    muted: 'rgba(0, 83, 253, 0.07)',
    mutedForeground: '#5B6475',
    popover: 'rgba(255, 255, 255, 0.92)',
    popoverForeground: '#111827',
    primary: '#0053FD',
    primaryForeground: '#FFFFFF',
    secondary: 'rgba(0, 83, 253, 0.08)',
    secondaryForeground: '#003EA8',
    accent: '#FFAC02',
    accentForeground: '#1A1200',
    border: 'rgba(0, 83, 253, 0.16)',
    input: 'rgba(0, 83, 253, 0.12)',
    ring: '#0053FD',
    midground: '#0053FD',
    composerRing: '#0053FD',
    destructive: '#C72E4D',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: 'rgba(255, 255, 255, 0.70)',
    sidebarBorder: 'rgba(0, 83, 253, 0.12)',
    userBubble: 'rgba(0, 83, 253, 0.08)',
    userBubbleBorder: 'rgba(0, 83, 253, 0.16)'
  },
  typography: {
    fontSans: '"Inter", ' + SYSTEM_SANS,
    fontMono: '"JetBrains Mono", ' + SYSTEM_MONO,
    fontUrl:
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap'
  }
}

/** Deep blue-violet with cool accents. Matches the dashboard midnight theme. */
export const midnightTheme: DesktopTheme = {
  name: 'midnight',
  label: 'Midnight',
  description: 'Deep blue-violet with cool accents',
  colors: {
    background: '#08081c',
    foreground: '#ddd6ff',
    card: '#0d0d28',
    cardForeground: '#ddd6ff',
    muted: '#13133a',
    mutedForeground: '#7c7ab0',
    popover: '#0f0f2e',
    popoverForeground: '#ddd6ff',
    primary: '#ddd6ff',
    primaryForeground: '#08081c',
    secondary: '#1a1a4a',
    secondaryForeground: '#c4bff0',
    accent: '#1a1a44',
    accentForeground: '#d0c8ff',
    border: '#1e1e52',
    input: '#1e1e52',
    ring: '#8b80e8',
    midground: '#8b80e8',
    destructive: '#b03060',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#06061a',
    sidebarBorder: '#12123a',
    userBubble: '#14143a',
    userBubbleBorder: '#242466'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap'
  }
}

/** Warm crimson and bronze — forge vibes. Matches the CLI ares skin. */
export const emberTheme: DesktopTheme = {
  name: 'ember',
  label: 'Ember',
  description: 'Warm crimson and bronze — forge vibes',
  colors: {
    background: '#160800',
    foreground: '#ffd8b0',
    card: '#1e0e04',
    cardForeground: '#ffd8b0',
    muted: '#2a1408',
    mutedForeground: '#aa7a56',
    popover: '#221008',
    popoverForeground: '#ffd8b0',
    primary: '#ffd8b0',
    primaryForeground: '#160800',
    secondary: '#341800',
    secondaryForeground: '#f0c090',
    accent: '#301600',
    accentForeground: '#e8c080',
    border: '#3a1c08',
    input: '#3a1c08',
    ring: '#d97316',
    midground: '#d97316',
    destructive: '#c43010',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#100600',
    sidebarBorder: '#2a1004',
    userBubble: '#2a1000',
    userBubbleBorder: '#4a2010'
  },
  typography: {
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap'
  }
}

/** Clean grayscale. Matches the CLI mono skin and dashboard mono theme. */
export const monoTheme: DesktopTheme = {
  name: 'mono',
  label: 'Mono',
  description: 'Clean grayscale — minimal and focused',
  colors: {
    background: '#0e0e0e',
    foreground: '#eaeaea',
    card: '#141414',
    cardForeground: '#eaeaea',
    muted: '#1e1e1e',
    mutedForeground: '#808080',
    popover: '#181818',
    popoverForeground: '#eaeaea',
    primary: '#eaeaea',
    primaryForeground: '#0e0e0e',
    secondary: '#262626',
    secondaryForeground: '#c8c8c8',
    accent: '#222222',
    accentForeground: '#d8d8d8',
    border: '#2a2a2a',
    input: '#2a2a2a',
    ring: '#9a9a9a',
    midground: '#9a9a9a',
    destructive: '#a84040',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#0a0a0a',
    sidebarBorder: '#202020',
    userBubble: '#1a1a1a',
    userBubbleBorder: '#363636'
  }
}

/** Neon green on black. Matches the CLI cyberpunk skin and dashboard theme. */
export const cyberpunkTheme: DesktopTheme = {
  name: 'cyberpunk',
  label: 'Cyberpunk',
  description: 'Neon green on black — matrix terminal',
  colors: {
    background: '#000a00',
    foreground: '#00ff41',
    card: '#001200',
    cardForeground: '#00ff41',
    muted: '#001a00',
    mutedForeground: '#1a8a30',
    popover: '#001000',
    popoverForeground: '#00ff41',
    primary: '#00ff41',
    primaryForeground: '#000a00',
    secondary: '#002800',
    secondaryForeground: '#00cc34',
    accent: '#002000',
    accentForeground: '#00e038',
    border: '#003000',
    input: '#003000',
    ring: '#00ff41',
    midground: '#00ff41',
    destructive: '#ff003c',
    destructiveForeground: '#000a00',
    sidebarBackground: '#000600',
    sidebarBorder: '#001800',
    userBubble: '#001400',
    userBubbleBorder: '#004800'
  },
  typography: {
    fontMono: `"Courier New", Courier, monospace, ${EMOJI_FALLBACK}`,
    fontSans: `"Courier New", Courier, monospace, ${EMOJI_FALLBACK}`
  }
}

/** Cool slate blue for developers. Matches the CLI slate skin. */
export const slateTheme: DesktopTheme = {
  name: 'slate',
  label: 'Slate',
  description: 'Cool slate blue — focused developer theme',
  colors: {
    background: '#0d1117',
    foreground: '#c9d1d9',
    card: '#161b22',
    cardForeground: '#c9d1d9',
    muted: '#21262d',
    mutedForeground: '#8b949e',
    popover: '#1c2128',
    popoverForeground: '#c9d1d9',
    primary: '#c9d1d9',
    primaryForeground: '#0d1117',
    secondary: '#2a3038',
    secondaryForeground: '#adb5bf',
    accent: '#1e2530',
    accentForeground: '#c0c8d0',
    border: '#30363d',
    input: '#30363d',
    ring: '#58a6ff',
    midground: '#58a6ff',
    destructive: '#cf4848',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#090d13',
    sidebarBorder: '#1c2228',
    userBubble: '#1e2a38',
    userBubbleBorder: '#2e4060'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const BUILTIN_THEMES: Record<string, DesktopTheme> = {
  'liquid-glass': liquidGlassTheme,
  'liquid-glass-light': liquidGlassLightTheme,
  nous: nousTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  slate: slateTheme
}

export const BUILTIN_THEME_LIST = Object.values(BUILTIN_THEMES)

/** Skin used when nothing is persisted or the persisted name is retired. */
export const DEFAULT_SKIN_NAME = 'liquid-glass'
