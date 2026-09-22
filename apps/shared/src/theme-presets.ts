/**
 * Raw palette table for every built-in Hermes theme preset — the single source
 * of truth shared by the desktop app (which layers OKLCH synthesis, terminal
 * palettes and typography on top) and the web dashboard (which projects each
 * preset down to its 3-slot background/midground/foreground model via
 * `webPresetFromShared`). Edit a preset's colours HERE; both surfaces follow.
 *
 * The palette-bearing presets (nous, github, catppuccin, everforest, solarized)
 * are forks of their VS Code originals converted by the desktop's
 * `buildThemeFromMarketplace`; re-convert from the upstream extension rather
 * than hand-editing hexes. `nous-alt` is first-party — do not re-derive it.
 */

/** Tailwind-style colour slots a preset carries (light palette, or the only palette). */
export interface ThemePresetColors {
  background: string
  foreground: string
  card: string
  cardForeground: string
  muted: string
  mutedForeground: string
  popover: string
  popoverForeground: string
  primary: string
  primaryForeground: string
  secondary: string
  secondaryForeground: string
  accent: string
  accentForeground: string
  border: string
  input: string
  /** Generic focus ring — buttons, inputs, etc. */
  ring: string
  /**
   * Brand-accent stroke — focus rings, streaming cursors, active session
   * pills, branded scrollbars, text selection. Falls back to `ring`.
   * Aliased to the DS `--midground` token.
   */
  midground?: string
  /** Auto-derived from `midground` luminance when omitted. */
  midgroundForeground?: string
  /** Composer outline / focus color. Falls back to `midground`. */
  composerRing?: string
  destructive: string
  destructiveForeground: string
  sidebarBackground?: string
  sidebarBorder?: string
  userBubble?: string
  userBubbleBorder?: string
}

export interface ThemePresetPalette {
  /** Light palette (also reused for dark when `darkColors` is omitted). */
  colors: ThemePresetColors
  /** Hand-tuned dark palette. Skins like `nous` ship one. */
  darkColors?: ThemePresetColors
}

const NOUS_ALT_BLUE = '#0053FD'
const NOUS_ALT_NAVY = '#1540B1'
const NOUS_ALT_CREAM = '#FFE6CB'

const nousAltTint = (pct: number) => `color-mix(in srgb, ${NOUS_ALT_BLUE} ${pct}%, #FFFFFF)`
const nousAltTintTransparent = (pct: number) => `color-mix(in srgb, ${NOUS_ALT_BLUE} ${pct}%, transparent)`

export const THEME_PRESET_PALETTES = {
  github: {
    colors: {
      background: '#ffffff',
      foreground: '#1f2328',
      card: '#f6f8fa',
      cardForeground: '#1f2328',
      muted: '#f6f6f6',
      mutedForeground: '#656d76',
      popover: '#ffffff',
      popoverForeground: '#1f2328',
      primary: '#196d31',
      primaryForeground: '#ffffff',
      secondary: '#dfebe2',
      secondaryForeground: '#1f2328',
      accent: '#e3ede6',
      accentForeground: '#1f2328',
      border: '#d0d7de',
      input: '#ffffff',
      ring: '#196d31',
      midground: '#196d31',
      midgroundForeground: '#ffffff',
      composerRing: '#196d31',
      destructive: '#cf222e',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#f6f8fa',
      sidebarBorder: '#d0d7de',
      userBubble: '#dbe7e2',
      userBubbleBorder: '#d0d7de'
    },
    darkColors: {
      background: '#0d1117',
      foreground: '#e6edf3',
      card: '#010409',
      cardForeground: '#e6edf3',
      muted: '#1a1e24',
      mutedForeground: '#7d8590',
      popover: '#161b22',
      popoverForeground: '#e6edf3',
      primary: '#4f9e5e',
      primaryForeground: '#ffffff',
      secondary: '#1f382b',
      secondaryForeground: '#e6edf3',
      accent: '#192a24',
      accentForeground: '#e6edf3',
      border: '#30363d',
      input: '#0d1117',
      ring: '#4f9e5e',
      midground: '#4f9e5e',
      midgroundForeground: '#ffffff',
      composerRing: '#4f9e5e',
      destructive: '#f85149',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#010409',
      sidebarBorder: '#30363d',
      userBubble: '#0f2018',
      userBubbleBorder: '#30363d'
    }
  },
  nous: {
    colors: {
      background: '#ffffff',
      foreground: '#1f2328',
      card: '#f6f8fa',
      cardForeground: '#1f2328',
      muted: '#f6f6f6',
      mutedForeground: '#656d76',
      popover: '#ffffff',
      popoverForeground: '#1f2328',
      primary: '#0053fd',
      primaryForeground: '#ffffff',
      secondary: '#deeaff',
      secondaryForeground: '#1f2328',
      accent: '#e3edff',
      accentForeground: '#1f2328',
      border: '#d0d7de',
      input: '#ffffff',
      ring: '#0053fd',
      midground: '#0053fd',
      midgroundForeground: '#ffffff',
      composerRing: '#0053fd',
      destructive: '#cf222e',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#f6f8fa',
      sidebarBorder: '#d0d7de',
      userBubble: '#dae7fd',
      userBubbleBorder: '#d0d7de'
    },
    darkColors: {
      background: '#0d1117',
      foreground: '#e6edf3',
      card: '#010409',
      cardForeground: '#e6edf3',
      muted: '#1a1e24',
      mutedForeground: '#7d8590',
      popover: '#161b22',
      popoverForeground: '#e6edf3',
      primary: '#4a84fe',
      primaryForeground: '#161616',
      secondary: '#1d2e4f',
      secondaryForeground: '#e6edf3',
      accent: '#17243a',
      accentForeground: '#e6edf3',
      border: '#30363d',
      input: '#0d1117',
      ring: '#4a84fe',
      midground: '#4a84fe',
      midgroundForeground: '#161616',
      composerRing: '#4a84fe',
      destructive: '#f85149',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#010409',
      sidebarBorder: '#30363d',
      userBubble: '#07162c',
      userBubbleBorder: '#30363d'
    }
  },
  catppuccin: {
    colors: {
      background: '#eff1f5',
      foreground: '#4c4f69',
      card: '#e6e9ef',
      cardForeground: '#4c4f69',
      muted: '#e8ebef',
      mutedForeground: '#4c4f69',
      popover: '#e6e9ef',
      popoverForeground: '#4c4f69',
      primary: '#6d2ebf',
      primaryForeground: '#ffffff',
      secondary: '#ddd6ed',
      secondaryForeground: '#4c4f69',
      accent: '#dfdaef',
      accentForeground: '#4c4f69',
      border: '#acb0be',
      input: '#ccd0da',
      ring: '#6d2ebf',
      midground: '#6d2ebf',
      midgroundForeground: '#ffffff',
      composerRing: '#6d2ebf',
      destructive: '#d20f39',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#e6e9ef',
      sidebarBorder: '#acb0be',
      userBubble: '#d7d3e9',
      userBubbleBorder: '#acb0be'
    },
    darkColors: {
      background: '#1e1e2e',
      foreground: '#cdd6f4',
      card: '#181825',
      cardForeground: '#cdd6f4',
      muted: '#29293a',
      mutedForeground: '#cdd6f4',
      popover: '#181825',
      popoverForeground: '#cdd6f4',
      primary: '#cba6f7',
      primaryForeground: '#ffffff',
      secondary: '#4e4466',
      secondaryForeground: '#cdd6f4',
      accent: '#3d3652',
      accentForeground: '#cdd6f4',
      border: '#585b70',
      input: '#313244',
      ring: '#cba6f7',
      midground: '#cba6f7',
      midgroundForeground: '#ffffff',
      composerRing: '#cba6f7',
      destructive: '#f38ba8',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#181825',
      sidebarBorder: '#585b70',
      userBubble: '#38324b',
      userBubbleBorder: '#585b70'
    }
  },
  everforest: {
    colors: {
      background: '#fdf6e3',
      foreground: '#5c6a72',
      card: '#fdf6e3',
      cardForeground: '#5c6a72',
      muted: '#f7f0de',
      mutedForeground: '#939f91',
      popover: '#fdf6e3',
      popoverForeground: '#5c6a72',
      primary: '#586b35',
      primaryForeground: '#ffffff',
      secondary: '#e6e3cb',
      secondaryForeground: '#5c6a72',
      accent: '#e9e5ce',
      accentForeground: '#5c6a72',
      border: '#fdf6e3',
      input: '#fdf6e3',
      ring: '#586b35',
      midground: '#586b35',
      midgroundForeground: '#ffffff',
      composerRing: '#586b35',
      destructive: '#f1706f',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#fdf6e3',
      sidebarBorder: '#fdf6e3',
      userBubble: '#e9e5ce',
      userBubbleBorder: '#fdf6e3'
    },
    darkColors: {
      background: '#2d353b',
      foreground: '#d3c6aa',
      card: '#2d353b',
      cardForeground: '#d3c6aa',
      muted: '#373e42',
      mutedForeground: '#859289',
      popover: '#2d353b',
      popoverForeground: '#d3c6aa',
      primary: '#a7c080',
      primaryForeground: '#ffffff',
      secondary: '#4f5c4e',
      secondaryForeground: '#d3c6aa',
      accent: '#434e47',
      accentForeground: '#d3c6aa',
      border: '#2d353b',
      input: '#2d353b',
      ring: '#a7c080',
      midground: '#a7c080',
      midgroundForeground: '#ffffff',
      composerRing: '#a7c080',
      destructive: '#da6362',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#2d353b',
      sidebarBorder: '#2d353b',
      userBubble: '#434e47',
      userBubbleBorder: '#2d353b'
    }
  },
  solarized: {
    colors: {
      background: '#fdf6e3',
      foreground: '#1f1f1f',
      card: '#d3cbb7',
      cardForeground: '#1f1f1f',
      muted: '#f4eddb',
      mutedForeground: '#9ca8a6',
      popover: '#eee8d5',
      popoverForeground: '#1f1f1f',
      primary: '#675e34',
      primaryForeground: '#ffffff',
      secondary: '#e8e1cb',
      secondaryForeground: '#1f1f1f',
      accent: '#ebe4ce',
      accentForeground: '#1f1f1f',
      border: '#ddd6c1',
      input: '#ddd6c1',
      ring: '#675e34',
      midground: '#675e34',
      midgroundForeground: '#ffffff',
      composerRing: '#675e34',
      destructive: '#e25563',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#eee8d5',
      sidebarBorder: '#ddd6c1',
      userBubble: '#c6bea7',
      userBubbleBorder: '#ddd6c1'
    },
    darkColors: {
      background: '#002b36',
      foreground: '#839496',
      card: '#002b36',
      cardForeground: '#839496',
      muted: '#08313c',
      mutedForeground: '#586e75',
      popover: '#001f26',
      popoverForeground: '#839496',
      primary: '#6ea1c4',
      primaryForeground: '#ffffff',
      secondary: '#1f4c5e',
      secondaryForeground: '#839496',
      accent: '#144050',
      accentForeground: '#839496',
      border: '#234751',
      input: '#073642',
      ring: '#6ea1c4',
      midground: '#6ea1c4',
      midgroundForeground: '#ffffff',
      composerRing: '#6ea1c4',
      destructive: '#e35957',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#001f26',
      sidebarBorder: '#234751',
      userBubble: '#144050',
      userBubbleBorder: '#234751'
    }
  },
  'nous-alt': {
    colors: {
      background: '#F8FAFF',
      foreground: '#17171A',
      card: '#FFFFFF',
      cardForeground: '#17171A',
      muted: nousAltTint(5),
      mutedForeground: '#666678',
      popover: '#FFFFFF',
      popoverForeground: '#17171A',
      primary: NOUS_ALT_BLUE,
      primaryForeground: '#FCFCFC',
      secondary: nousAltTint(7),
      secondaryForeground: '#242432',
      accent: nousAltTint(10),
      accentForeground: '#202030',
      border: nousAltTintTransparent(22),
      input: nousAltTintTransparent(30),
      ring: NOUS_ALT_BLUE,
      midground: NOUS_ALT_BLUE,
      composerRing: NOUS_ALT_BLUE,
      destructive: '#C72E4D',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F3F7FF',
      sidebarBorder: nousAltTintTransparent(18),
      userBubble: nousAltTint(6),
      userBubbleBorder: nousAltTintTransparent(24)
    },
    darkColors: {
      background: '#0D2F86',
      foreground: NOUS_ALT_CREAM,
      card: '#12378F',
      cardForeground: NOUS_ALT_CREAM,
      muted: '#183F9A',
      mutedForeground: '#B5C7F3',
      popover: '#123A96',
      popoverForeground: NOUS_ALT_CREAM,
      primary: NOUS_ALT_CREAM,
      primaryForeground: '#0D2F86',
      secondary: '#1B45A4',
      secondaryForeground: '#E0E8FF',
      accent: NOUS_ALT_NAVY,
      accentForeground: '#F0F4FF',
      border: '#3158AD',
      input: '#0B2566',
      ring: NOUS_ALT_CREAM,
      midground: NOUS_ALT_BLUE,
      composerRing: NOUS_ALT_CREAM,
      destructive: '#C0473A',
      destructiveForeground: '#FEF2F2',
      sidebarBackground: '#09286F',
      sidebarBorder: '#234A9C',
      userBubble: '#143B91',
      userBubbleBorder: '#3A63BD'
    }
  },
  midnight: {
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
    }
  },
  ember: {
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
    }
  },
  mono: {
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
  },
  cyberpunk: {
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
    }
  },
  slate: {
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
    }
  },
  jarvis: {
    colors: {
      background: '#040d1a',
      foreground: '#e0f7fa',
      card: '#07162c',
      cardForeground: '#e0f7fa',
      muted: '#0b2344',
      mutedForeground: '#7dd3fc',
      popover: '#07162c',
      popoverForeground: '#e0f7fa',
      primary: '#00f0ff',
      primaryForeground: '#040d1a',
      secondary: '#0c2e56',
      secondaryForeground: '#38bdf8',
      accent: '#0e3a6c',
      accentForeground: '#00f0ff',
      border: '#0088cc',
      input: '#07162c',
      ring: '#00f0ff',
      midground: '#00f0ff',
      midgroundForeground: '#040d1a',
      composerRing: '#00f0ff',
      destructive: '#ff3366',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#020813',
      sidebarBorder: '#0088cc',
      userBubble: '#092548',
      userBubbleBorder: '#00f0ff'
    },
    darkColors: {
      background: '#040d1a',
      foreground: '#e0f7fa',
      card: '#07162c',
      cardForeground: '#e0f7fa',
      muted: '#0b2344',
      mutedForeground: '#7dd3fc',
      popover: '#07162c',
      popoverForeground: '#e0f7fa',
      primary: '#00f0ff',
      primaryForeground: '#040d1a',
      secondary: '#0c2e56',
      secondaryForeground: '#38bdf8',
      accent: '#0e3a6c',
      accentForeground: '#00f0ff',
      border: '#0088cc',
      input: '#07162c',
      ring: '#00f0ff',
      midground: '#00f0ff',
      midgroundForeground: '#040d1a',
      composerRing: '#00f0ff',
      destructive: '#ff3366',
      destructiveForeground: '#ffffff',
      sidebarBackground: '#020813',
      sidebarBorder: '#0088cc',
      userBubble: '#092548',
      userBubbleBorder: '#00f0ff'
    }
  }
} satisfies Record<string, ThemePresetPalette>

export type ThemePresetName = keyof typeof THEME_PRESET_PALETTES
