/**
 * Built-in desktop themes.
 *
 * Names match the CLI skins and dashboard theme presets so users get
 * a consistent visual identity across surfaces.
 *
 * Add new themes here — no code changes needed elsewhere.
 */

import type { DesktopTheme, DesktopThemeTypography } from './types'

// ---------------------------------------------------------------------------
// Shared defaults
// ---------------------------------------------------------------------------

const SYSTEM_SANS =
  'ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, system-ui, sans-serif'

const SYSTEM_MONO =
  'ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Code", Menlo, Monaco, Consolas, "Liberation Mono", monospace'

export const DEFAULT_TYPOGRAPHY: DesktopThemeTypography = {
  fontSans: SYSTEM_SANS,
  fontMono: SYSTEM_MONO
}

// ---------------------------------------------------------------------------
// Built-in themes
// ---------------------------------------------------------------------------

/** Hermes light — clean neutral canvas with metakina green accents. */
export const nousLightTheme: DesktopTheme = {
  name: 'nous-light',
  label: 'Hermes Light',
  description: 'Cool white with metakina green accents — quiet and operational',
  colors: {
    background: '#F7FAF8',
    foreground: '#18201C',
    card: '#FFFFFF',
    cardForeground: '#18201C',
    muted: '#EDF3EF',
    mutedForeground: '#66736B',
    popover: '#FFFFFF',
    popoverForeground: '#18201C',
    primary: '#31B768',
    primaryForeground: '#ffffff',
    secondary: '#EAF2EE',
    secondaryForeground: '#1E3128',
    accent: '#E8F4ED',
    accentForeground: '#1E3128',
    border: '#D9E4DE',
    input: '#D3E0D8',
    ring: '#2FAE63',
    midground: '#31B768',
    destructive: '#b94a3a',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#F1F6F3',
    sidebarBorder: '#D9E4DE',
    userBubble: '#EAF2EE',
    userBubbleBorder: '#D9E4DE'
  }
}

/** Optional Hermes gold skin for people who want the classic TUI accent. */
export const hermesGoldTheme: DesktopTheme = {
  name: 'gold',
  label: 'Gold',
  description: 'Classic Hermes gold accent',
  colors: {
    ...nousLightTheme.colors,
    primary: '#d4af37',
    primaryForeground: '#1a1404',
    secondary: '#f6efd5',
    secondaryForeground: '#5a4310',
    accent: '#fbf3d4',
    accentForeground: '#5a4310',
    ring: '#d4af37',
    midground: '#d4af37',
    userBubble: '#f6efd5'
  }
}

const NOUS_LENS_BLUE = '#0053FD'

/** Nous — bright white with electric blue from the NousNet identity system. */
export const nousTheme: DesktopTheme = {
  name: 'nous',
  label: 'Nous',
  description: 'Design-system white with electric Nous blue and subtle grain',
  colors: {
    background: '#FFFFFF',
    foreground: '#17171A',
    card: '#FFFFFF',
    cardForeground: '#17171A',
    muted: `color-mix(in srgb, ${NOUS_LENS_BLUE} 5%, #FFFFFF)`,
    mutedForeground: '#666678',
    popover: '#FFFFFF',
    popoverForeground: '#17171A',
    primary: NOUS_LENS_BLUE,
    primaryForeground: '#FFFFFF',
    secondary: `color-mix(in srgb, ${NOUS_LENS_BLUE} 7%, #FFFFFF)`,
    secondaryForeground: '#242432',
    accent: `color-mix(in srgb, ${NOUS_LENS_BLUE} 10%, #FFFFFF)`,
    accentForeground: '#202030',
    border: `color-mix(in srgb, ${NOUS_LENS_BLUE} 22%, transparent)`,
    input: `color-mix(in srgb, ${NOUS_LENS_BLUE} 30%, transparent)`,
    ring: NOUS_LENS_BLUE,
    midground: NOUS_LENS_BLUE,
    destructive: '#C72E4D',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: `color-mix(in srgb, ${NOUS_LENS_BLUE} 2.5%, #FFFFFF)`,
    sidebarBorder: `color-mix(in srgb, ${NOUS_LENS_BLUE} 18%, transparent)`,
    userBubble: `color-mix(in srgb, ${NOUS_LENS_BLUE} 6%, #FFFFFF)`,
    userBubbleBorder: `color-mix(in srgb, ${NOUS_LENS_BLUE} 24%, transparent)`
  },
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: `"Courier Prime", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap'
  }
}

/** Classic Hermes dark teal. */
export const defaultTheme: DesktopTheme = {
  name: 'default',
  label: 'Hermes Teal',
  description: 'Classic dark teal — the canonical Hermes look',
  colors: {
    background: '#0d1a1a',
    foreground: '#f0e8d8',
    card: '#111f1f',
    cardForeground: '#f0e8d8',
    muted: '#172828',
    mutedForeground: '#8aada6',
    popover: '#142222',
    popoverForeground: '#f0e8d8',
    primary: '#f0e8d8',
    primaryForeground: '#0d1a1a',
    secondary: '#1e3030',
    secondaryForeground: '#c8ddd8',
    accent: '#1b2e2e',
    accentForeground: '#e0d4c0',
    border: '#1e3232',
    input: '#1e3232',
    ring: '#6bbfb5',
    midground: '#6bbfb5',
    destructive: '#c0473a',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#0a1616',
    sidebarBorder: '#172424',
    userBubble: '#1a2e2e',
    userBubbleBorder: '#2a4a44'
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
    fontMono: `"Courier New", Courier, monospace`,
    fontSans: `"Courier New", Courier, monospace`
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

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const BUILTIN_THEMES: Record<string, DesktopTheme> = {
  'nous-light': nousLightTheme,
  default: defaultTheme,
  nous: nousTheme,
  gold: hermesGoldTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  slate: slateTheme
}

export const BUILTIN_THEME_LIST = Object.values(BUILTIN_THEMES)
