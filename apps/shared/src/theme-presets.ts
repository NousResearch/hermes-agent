/**
 * Raw palette table for every built-in Unitalk theme preset — the single source
 * of truth shared by the desktop app (which layers OKLCH synthesis, terminal
 * palettes and typography on top) and the web dashboard (which projects each
 * preset down to its 3-slot background/midground/foreground model via
 * `webPresetFromShared`). Edit a preset's colours HERE; both surfaces follow.
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
  /** Hand-tuned dark palette. */
  darkColors?: ThemePresetColors
}

export const THEME_PRESET_PALETTES = {
  /** Microsoft — inspired by Microsoft 365/Windows. */
  microsoft: {
    colors: {
      background: '#FFFFFF',
      foreground: '#242424',
      card: '#F5F5F5',
      cardForeground: '#242424',
      muted: '#F0F0F0',
      mutedForeground: '#616161',
      popover: '#FFFFFF',
      popoverForeground: '#242424',
      primary: '#0078D4',
      primaryForeground: '#FFFFFF',
      secondary: '#DEECF9',
      secondaryForeground: '#242424',
      accent: '#E5F0FA',
      accentForeground: '#242424',
      border: '#E0E0E0',
      input: '#FFFFFF',
      ring: '#0078D4',
      midground: '#0078D4',
      midgroundForeground: '#FFFFFF',
      composerRing: '#0078D4',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F5F5F5',
      sidebarBorder: '#E0E0E0',
      userBubble: '#DEECF9',
      userBubbleBorder: '#E0E0E0'
    },
    darkColors: {
      background: '#1E1E1E',
      foreground: '#E8E8E8',
      card: '#252525',
      cardForeground: '#E8E8E8',
      muted: '#2D2D2D',
      mutedForeground: '#999999',
      popover: '#252525',
      popoverForeground: '#E8E8E8',
      primary: '#4DA3E0',
      primaryForeground: '#FFFFFF',
      secondary: '#1A3A5C',
      secondaryForeground: '#E8E8E8',
      accent: '#1E3350',
      accentForeground: '#E8E8E8',
      border: '#3E3E3E',
      input: '#252525',
      ring: '#4DA3E0',
      midground: '#4DA3E0',
      midgroundForeground: '#FFFFFF',
      composerRing: '#4DA3E0',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#1A1A1A',
      sidebarBorder: '#3E3E3E',
      userBubble: '#1A3A5C',
      userBubbleBorder: '#3E3E3E'
    }
  },

  /** Apple — inspired by macOS/iOS. */
  apple: {
    colors: {
      background: '#F5F5F7',
      foreground: '#1D1D1F',
      card: '#FFFFFF',
      cardForeground: '#1D1D1F',
      muted: '#F0F0F2',
      mutedForeground: '#86868B',
      popover: '#FFFFFF',
      popoverForeground: '#1D1D1F',
      primary: '#007AFF',
      primaryForeground: '#FFFFFF',
      secondary: '#E8F0FE',
      secondaryForeground: '#1D1D1F',
      accent: '#EBF3FE',
      accentForeground: '#1D1D1F',
      border: '#D2D2D7',
      input: '#FFFFFF',
      ring: '#007AFF',
      midground: '#007AFF',
      midgroundForeground: '#FFFFFF',
      composerRing: '#007AFF',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F0F0F2',
      sidebarBorder: '#D2D2D7',
      userBubble: '#E8F0FE',
      userBubbleBorder: '#D2D2D7'
    },
    darkColors: {
      background: '#1C1C1E',
      foreground: '#F5F5F7',
      card: '#2C2C2E',
      cardForeground: '#F5F5F7',
      muted: '#3A3A3C',
      mutedForeground: '#98989D',
      popover: '#2C2C2E',
      popoverForeground: '#F5F5F7',
      primary: '#5E9EFF',
      primaryForeground: '#FFFFFF',
      secondary: '#1E3A5F',
      secondaryForeground: '#F5F5F7',
      accent: '#1C3150',
      accentForeground: '#F5F5F7',
      border: '#38383A',
      input: '#2C2C2E',
      ring: '#5E9EFF',
      midground: '#5E9EFF',
      midgroundForeground: '#FFFFFF',
      composerRing: '#5E9EFF',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#161618',
      sidebarBorder: '#38383A',
      userBubble: '#1E3A5F',
      userBubbleBorder: '#38383A'
    }
  },

  /** Google — inspired by Google Workspace / Material Design. */
  google: {
    colors: {
      background: '#FFFFFF',
      foreground: '#202124',
      card: '#F8F9FA',
      cardForeground: '#202124',
      muted: '#F1F3F4',
      mutedForeground: '#5F6368',
      popover: '#FFFFFF',
      popoverForeground: '#202124',
      primary: '#4285F4',
      primaryForeground: '#FFFFFF',
      secondary: '#E8F0FE',
      secondaryForeground: '#202124',
      accent: '#E0EBFC',
      accentForeground: '#202124',
      border: '#DADCE0',
      input: '#FFFFFF',
      ring: '#4285F4',
      midground: '#4285F4',
      midgroundForeground: '#FFFFFF',
      composerRing: '#4285F4',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F8F9FA',
      sidebarBorder: '#DADCE0',
      userBubble: '#E8F0FE',
      userBubbleBorder: '#DADCE0'
    },
    darkColors: {
      background: '#121212',
      foreground: '#E8EAED',
      card: '#1E1E1E',
      cardForeground: '#E8EAED',
      muted: '#2D2D2D',
      mutedForeground: '#9AA0A6',
      popover: '#1E1E1E',
      popoverForeground: '#E8EAED',
      primary: '#8AB4F8',
      primaryForeground: '#121212',
      secondary: '#1A3A5C',
      secondaryForeground: '#E8EAED',
      accent: '#1E3350',
      accentForeground: '#E8EAED',
      border: '#3C4043',
      input: '#1E1E1E',
      ring: '#8AB4F8',
      midground: '#8AB4F8',
      midgroundForeground: '#121212',
      composerRing: '#8AB4F8',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#0D0D0D',
      sidebarBorder: '#3C4043',
      userBubble: '#1A3A5C',
      userBubbleBorder: '#3C4043'
    }
  },

  /** ChatGPT — inspired by ChatGPT interface. */
  chatgpt: {
    colors: {
      background: '#FFFFFF',
      foreground: '#343541',
      card: '#F7F7F8',
      cardForeground: '#343541',
      muted: '#ECECF1',
      mutedForeground: '#8E8EA0',
      popover: '#FFFFFF',
      popoverForeground: '#343541',
      primary: '#10A37F',
      primaryForeground: '#FFFFFF',
      secondary: '#DCF3EB',
      secondaryForeground: '#343541',
      accent: '#E8F8F3',
      accentForeground: '#343541',
      border: '#E5E5E5',
      input: '#FFFFFF',
      ring: '#10A37F',
      midground: '#10A37F',
      midgroundForeground: '#FFFFFF',
      composerRing: '#10A37F',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F7F7F8',
      sidebarBorder: '#E5E5E5',
      userBubble: '#DCF3EB',
      userBubbleBorder: '#E5E5E5'
    },
    darkColors: {
      background: '#212121',
      foreground: '#ECECF1',
      card: '#2D2D2D',
      cardForeground: '#ECECF1',
      muted: '#3E3E3E',
      mutedForeground: '#8E8EA0',
      popover: '#2D2D2D',
      popoverForeground: '#ECECF1',
      primary: '#3CC99E',
      primaryForeground: '#212121',
      secondary: '#1A3D33',
      secondaryForeground: '#ECECF1',
      accent: '#1E3A32',
      accentForeground: '#ECECF1',
      border: '#4D4D4D',
      input: '#2D2D2D',
      ring: '#3CC99E',
      midground: '#3CC99E',
      midgroundForeground: '#212121',
      composerRing: '#3CC99E',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#171717',
      sidebarBorder: '#4D4D4D',
      userBubble: '#1A3D33',
      userBubbleBorder: '#4D4D4D'
    }
  },

  /** Claude — inspired by Claude/Anthropic interface. */
  claude: {
    colors: {
      background: '#F9F7F2',
      foreground: '#2D2A26',
      card: '#FFFFFF',
      cardForeground: '#2D2A26',
      muted: '#F3F0EB',
      mutedForeground: '#8B8278',
      popover: '#FFFFFF',
      popoverForeground: '#2D2A26',
      primary: '#D97706',
      primaryForeground: '#FFFFFF',
      secondary: '#FEF3E0',
      secondaryForeground: '#2D2A26',
      accent: '#FDF0D8',
      accentForeground: '#2D2A26',
      border: '#E6DFD2',
      input: '#FFFFFF',
      ring: '#D97706',
      midground: '#D97706',
      midgroundForeground: '#FFFFFF',
      composerRing: '#D97706',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F5F2EC',
      sidebarBorder: '#E6DFD2',
      userBubble: '#FEF3E0',
      userBubbleBorder: '#E6DFD2'
    },
    darkColors: {
      background: '#1E1B18',
      foreground: '#F0ECE4',
      card: '#2A2622',
      cardForeground: '#F0ECE4',
      muted: '#3D3832',
      mutedForeground: '#9E9588',
      popover: '#2A2622',
      popoverForeground: '#F0ECE4',
      primary: '#F59E0B',
      primaryForeground: '#1E1B18',
      secondary: '#3D2A10',
      secondaryForeground: '#F0ECE4',
      accent: '#382818',
      accentForeground: '#F0ECE4',
      border: '#3D3832',
      input: '#2A2622',
      ring: '#F59E0B',
      midground: '#F59E0B',
      midgroundForeground: '#1E1B18',
      composerRing: '#F59E0B',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#151310',
      sidebarBorder: '#3D3832',
      userBubble: '#3D2A10',
      userBubbleBorder: '#3D3832'
    }
  },

  /** Grok — inspired by xAI Grok. Stark black-on-white / white-on-black. */
  grok: {
    colors: {
      background: '#FFFFFF',
      foreground: '#000000',
      card: '#F5F5F5',
      cardForeground: '#000000',
      muted: '#F0F0F0',
      mutedForeground: '#707070',
      popover: '#FFFFFF',
      popoverForeground: '#000000',
      primary: '#000000',
      primaryForeground: '#FFFFFF',
      secondary: '#E5E5E5',
      secondaryForeground: '#000000',
      accent: '#EBEBEB',
      accentForeground: '#000000',
      border: '#D0D0D0',
      input: '#FFFFFF',
      ring: '#000000',
      midground: '#000000',
      midgroundForeground: '#FFFFFF',
      composerRing: '#000000',
      destructive: '#DC2626',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F5F5F5',
      sidebarBorder: '#D0D0D0',
      userBubble: '#E5E5E5',
      userBubbleBorder: '#D0D0D0'
    },
    darkColors: {
      background: '#000000',
      foreground: '#FFFFFF',
      card: '#111111',
      cardForeground: '#FFFFFF',
      muted: '#1A1A1A',
      mutedForeground: '#888888',
      popover: '#111111',
      popoverForeground: '#FFFFFF',
      primary: '#FFFFFF',
      primaryForeground: '#000000',
      secondary: '#222222',
      secondaryForeground: '#FFFFFF',
      accent: '#1C1C1C',
      accentForeground: '#FFFFFF',
      border: '#333333',
      input: '#111111',
      ring: '#FFFFFF',
      midground: '#FFFFFF',
      midgroundForeground: '#000000',
      composerRing: '#FFFFFF',
      destructive: '#DC2626',
      destructiveForeground: '#FCA5A5',
      sidebarBackground: '#050505',
      sidebarBorder: '#333333',
      userBubble: '#222222',
      userBubbleBorder: '#333333'
    }
  },

  /** GitHub — keeps the upstream palette unmodified. */
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

  forest: {
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

  /** Unitalk — the canonical brand theme. Light: warm paper (#F3EFE6) with
   *  magenta accents (#D10E63). Dark: deep ink (#1C1A17) with the same
   *  accent colour lifted for readability. Both modes share the Unitalk
   *  identity: warm, professional, readable. */
  unitalk: {
    colors: {
      background: '#F3EFE6',
      foreground: '#1C1A17',
      card: '#FFFFFF',
      cardForeground: '#1C1A17',
      muted: '#EDE8DD',
      mutedForeground: '#7A7468',
      popover: '#FFFFFF',
      popoverForeground: '#1C1A17',
      primary: '#D10E63',
      primaryForeground: '#FFFFFF',
      secondary: '#F8E8EF',
      secondaryForeground: '#1C1A17',
      accent: '#F5D9E5',
      accentForeground: '#1C1A17',
      border: '#D8D0C2',
      input: '#FFFFFF',
      ring: '#D10E63',
      midground: '#D10E63',
      midgroundForeground: '#FFFFFF',
      composerRing: '#D10E63',
      destructive: '#C62828',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#F8F4ED',
      sidebarBorder: '#D8D0C2',
      userBubble: '#F5D9E5',
      userBubbleBorder: '#D8D0C2'
    },
    darkColors: {
      background: '#1C1A17',
      foreground: '#F3EFE6',
      card: '#282520',
      cardForeground: '#F3EFE6',
      muted: '#2E2B25',
      mutedForeground: '#9E9888',
      popover: '#24211C',
      popoverForeground: '#F3EFE6',
      primary: '#E83E7E',
      primaryForeground: '#FFFFFF',
      secondary: '#3D1A2A',
      secondaryForeground: '#F3EFE6',
      accent: '#2E1D25',
      accentForeground: '#F3EFE6',
      border: '#3D3930',
      input: '#2E2B25',
      ring: '#E83E7E',
      midground: '#E83E7E',
      midgroundForeground: '#FFFFFF',
      composerRing: '#E83E7E',
      destructive: '#EF5350',
      destructiveForeground: '#FFFFFF',
      sidebarBackground: '#161410',
      sidebarBorder: '#3D3930',
      userBubble: '#2E1D25',
      userBubbleBorder: '#3D3930'
    }
  }
} satisfies Record<string, ThemePresetPalette>

export type ThemePresetName = keyof typeof THEME_PRESET_PALETTES