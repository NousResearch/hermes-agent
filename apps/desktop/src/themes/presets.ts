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

export const roseTheme: DesktopTheme = {
  name: 'rose',
  label: 'Rose',
  description: 'Dusky rose and warm sand accents — elegant and calm',
  colors: {
    background: '#29151e',
    foreground: '#ffdce5',
    card: '#351c27',
    cardForeground: '#ffdce5',
    muted: '#432332',
    mutedForeground: '#b28e9b',
    popover: '#351c27',
    popoverForeground: '#ffdce5',
    primary: '#ff9ebb',
    primaryForeground: '#29151e',
    secondary: '#4d2839',
    secondaryForeground: '#ffdce5',
    accent: '#3e202e',
    accentForeground: '#ffb3c6',
    border: '#4d2839',
    input: '#4d2839',
    ring: '#ff9ebb',
    midground: '#ff9ebb',
    destructive: '#a84054',
    destructiveForeground: '#fef2f2',
    sidebarBackground: '#1c0e14',
    sidebarBorder: '#321924',
    userBubble: '#381d29',
    userBubbleBorder: '#5e3246'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const forestTheme: DesktopTheme = {
  name: 'forest',
  label: 'Forest',
  description: 'Deep mossy green with warm gold and sage accents',
  colors: {
    background: '#122419',
    foreground: '#e2f3e8',
    card: '#1c3325',
    cardForeground: '#e2f3e8',
    muted: '#243f30',
    mutedForeground: '#8ba795',
    popover: '#1c3325',
    popoverForeground: '#e2f3e8',
    primary: '#60b37e',
    primaryForeground: '#122419',
    secondary: '#2c4d3b',
    secondaryForeground: '#e2f3e8',
    accent: '#1f3c2b',
    accentForeground: '#86cfa2',
    border: '#2c4d3b',
    input: '#2c4d3b',
    ring: '#60b37e',
    midground: '#60b37e',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#0b1710',
    sidebarBorder: '#1e3327',
    userBubble: '#244532',
    userBubbleBorder: '#3d6e52'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const oceanTheme: DesktopTheme = {
  name: 'ocean',
  label: 'Ocean',
  description: 'Deep abyss blue with bioluminescent sky accents',
  colors: {
    background: '#0c1b2f',
    foreground: '#d2e5ff',
    card: '#142944',
    cardForeground: '#d2e5ff',
    muted: '#1c3555',
    mutedForeground: '#7fa2ce',
    popover: '#142944',
    popoverForeground: '#d2e5ff',
    primary: '#38bdf8',
    primaryForeground: '#0c1b2f',
    secondary: '#204066',
    secondaryForeground: '#d2e5ff',
    accent: '#183454',
    accentForeground: '#7dd3fc',
    border: '#204066',
    input: '#204066',
    ring: '#38bdf8',
    midground: '#38bdf8',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#071221',
    sidebarBorder: '#142c4b',
    userBubble: '#20436c',
    userBubbleBorder: '#366ba7'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const draculaTheme: DesktopTheme = {
  name: 'dracula',
  label: 'Dracula',
  description: 'Classic dark vampire palette with gothic pink and purple',
  colors: {
    background: '#222133',
    foreground: '#f8f8f2',
    card: '#2c2b42',
    cardForeground: '#f8f8f2',
    muted: '#363550',
    mutedForeground: '#6272a4',
    popover: '#2c2b42',
    popoverForeground: '#f8f8f2',
    primary: '#ff79c6',
    primaryForeground: '#222133',
    secondary: '#363550',
    secondaryForeground: '#f8f8f2',
    accent: '#bd93f9',
    accentForeground: '#ff79c6',
    border: '#434164',
    input: '#434164',
    ring: '#bd93f9',
    midground: '#bd93f9',
    destructive: '#ff5555',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#171624',
    sidebarBorder: '#32314a',
    userBubble: '#383655',
    userBubbleBorder: '#5c5a89'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const nordTheme: DesktopTheme = {
  name: 'nord',
  label: 'Nord',
  description: 'Calm arctic slate and polar night shades with ice blue accents',
  colors: {
    background: '#2b3342',
    foreground: '#eceff4',
    card: '#353e4f',
    cardForeground: '#eceff4',
    muted: '#3f4a5e',
    mutedForeground: '#d8dee9',
    popover: '#353e4f',
    popoverForeground: '#eceff4',
    primary: '#88c0d0',
    primaryForeground: '#2b3342',
    secondary: '#3f4a5e',
    secondaryForeground: '#d8dee9',
    accent: '#81a1c1',
    accentForeground: '#8fbcbb',
    border: '#4c5970',
    input: '#4c5970',
    ring: '#88c0d0',
    midground: '#88c0d0',
    destructive: '#bf616a',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#1f2530',
    sidebarBorder: '#303a4b',
    userBubble: '#434f64',
    userBubbleBorder: '#586884'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const sunsetTheme: DesktopTheme = {
  name: 'sunset',
  label: 'Sunset',
  description: 'Deep dusk violet with vibrant coral and pink twilight accents',
  colors: {
    background: '#22122c',
    foreground: '#fcdff0',
    card: '#2e1a3b',
    cardForeground: '#fcdff0',
    muted: '#3a2149',
    mutedForeground: '#ab8ba9',
    popover: '#2e1a3b',
    popoverForeground: '#fcdff0',
    primary: '#ff7e67',
    primaryForeground: '#22122c',
    secondary: '#432655',
    secondaryForeground: '#fcdff0',
    accent: '#3e224e',
    accentForeground: '#ff5c8a',
    border: '#4f2d64',
    input: '#4f2d64',
    ring: '#ff7e67',
    midground: '#ff7e67',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#150a1b',
    sidebarBorder: '#381c47',
    userBubble: '#4a2b5e',
    userBubbleBorder: '#7a479b'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const solariaTheme: DesktopTheme = {
  name: 'solaria',
  label: 'Solaria',
  description: 'Rich solar charcoal with desert gold and bronze accents',
  colors: {
    background: '#251f15',
    foreground: '#fcedc2',
    card: '#31291d',
    cardForeground: '#fcedc2',
    muted: '#3d3325',
    mutedForeground: '#a3977c',
    popover: '#31291d',
    popoverForeground: '#fcedc2',
    primary: '#f59e0b',
    primaryForeground: '#251f15',
    secondary: '#493d2d',
    secondaryForeground: '#fcedc2',
    accent: '#382f22',
    accentForeground: '#fbbf24',
    border: '#493d2d',
    input: '#493d2d',
    ring: '#f59e0b',
    midground: '#f59e0b',
    destructive: '#b91c1c',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#19150e',
    sidebarBorder: '#382e21',
    userBubble: '#413627',
    userBubbleBorder: '#6d5a41'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const nebulaTheme: DesktopTheme = {
  name: 'nebula',
  label: 'Nebula',
  description: 'Deep cosmic indigo with lavender and hot nebula pink',
  colors: {
    background: '#161036',
    foreground: '#e7dfff',
    card: '#22194a',
    cardForeground: '#e7dfff',
    muted: '#2c225c',
    mutedForeground: '#8f83cc',
    popover: '#22194a',
    popoverForeground: '#e7dfff',
    primary: '#c084fc',
    primaryForeground: '#161036',
    secondary: '#362a70',
    secondaryForeground: '#e7dfff',
    accent: '#2b1f5e',
    accentForeground: '#f472b6',
    border: '#44348a',
    input: '#44348a',
    ring: '#c084fc',
    midground: '#c084fc',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#0f0a25',
    sidebarBorder: '#332769',
    userBubble: '#3c2e7b',
    userBubbleBorder: '#634dcb'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const gruvboxTheme: DesktopTheme = {
  name: 'gruvbox',
  label: 'Gruvbox',
  description: 'Retro warm-toned dark palette with pastel amber and green',
  colors: {
    background: '#262621',
    foreground: '#ebdbb2',
    card: '#32322a',
    cardForeground: '#ebdbb2',
    muted: '#3c3c33',
    mutedForeground: '#a89984',
    popover: '#32322a',
    popoverForeground: '#ebdbb2',
    primary: '#fabd2f',
    primaryForeground: '#262621',
    secondary: '#45453a',
    secondaryForeground: '#ebdbb2',
    accent: '#b8bb26',
    accentForeground: '#fabd2f',
    border: '#45453a',
    input: '#45453a',
    ring: '#fabd2f',
    midground: '#fabd2f',
    destructive: '#fb4934',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#1c1c18',
    sidebarBorder: '#32322a',
    userBubble: '#3c3c33',
    userBubbleBorder: '#5b5b4e'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const espressoTheme: DesktopTheme = {
  name: 'espresso',
  label: 'Espresso',
  description: 'Rich dark espresso roast with warm crema and taupe cream accents',
  colors: {
    background: '#231a16',
    foreground: '#f4edea',
    card: '#30241e',
    cardForeground: '#f4edea',
    muted: '#3a2c26',
    mutedForeground: '#a6958d',
    popover: '#30241e',
    popoverForeground: '#f4edea',
    primary: '#d4a373',
    primaryForeground: '#231a16',
    secondary: '#42322a',
    secondaryForeground: '#f4edea',
    accent: '#352822',
    accentForeground: '#e6ccb2',
    border: '#4b3930',
    input: '#4b3930',
    ring: '#d4a373',
    midground: '#d4a373',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#18110f',
    sidebarBorder: '#3a2c26',
    userBubble: '#42322a',
    userBubbleBorder: '#6b5143'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const sageTheme: DesktopTheme = {
  name: 'sage',
  label: 'Sage',
  description: 'Dark slate eucalyptus with calming sage and mint accents',
  colors: {
    background: '#182622',
    foreground: '#e2edea',
    card: '#243832',
    cardForeground: '#e2edea',
    muted: '#2d453e',
    mutedForeground: '#8fa8a1',
    popover: '#243832',
    popoverForeground: '#e2edea',
    primary: '#9ac2b6',
    primaryForeground: '#182622',
    secondary: '#3b5c53',
    secondaryForeground: '#e2edea',
    accent: '#314c44',
    accentForeground: '#a9d4c8',
    border: '#3b5c53',
    input: '#3b5c53',
    ring: '#9ac2b6',
    midground: '#9ac2b6',
    destructive: '#a84040',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#111a17',
    sidebarBorder: '#293f39',
    userBubble: '#3b5c53',
    userBubbleBorder: '#5b8e80'
  },
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const BUILTIN_THEMES: Record<string, DesktopTheme> = {
  nous: nousTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  slate: slateTheme,
  rose: roseTheme,
  forest: forestTheme,
  ocean: oceanTheme,
  dracula: draculaTheme,
  nord: nordTheme,
  sunset: sunsetTheme,
  solaria: solariaTheme,
  nebula: nebulaTheme,
  gruvbox: gruvboxTheme,
  espresso: espressoTheme,
  sage: sageTheme
}

export const BUILTIN_THEME_LIST = Object.values(BUILTIN_THEMES)

/** Skin used when nothing is persisted or the persisted name is retired. */
export const DEFAULT_SKIN_NAME = 'nous'
