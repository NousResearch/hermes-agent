import type { DesktopTerminalPalette, DesktopTheme, DesktopThemeColors } from './types'

// Runtime-only provenance: a user-installed theme named one-dark is not our preset.
// Object spreads preserve the symbol; JSON deliberately does not persist it.
const ONE_DARK = Symbol('one-dark-pro')
type OneDarkTheme = DesktopTheme & { [ONE_DARK]?: boolean }

export const isOneDark = (theme: DesktopTheme): boolean => (theme as OneDarkTheme)[ONE_DARK] === true

export const themeAccent = (theme: DesktopTheme): string =>
  isOneDark(theme) ? (theme.colors.midground ?? theme.colors.accent) : theme.colors.primary

const COLORS: DesktopThemeColors = {
  background: '#282c34',
  foreground: '#abb2bf',
  card: '#21252b',
  cardForeground: '#abb2bf',
  popover: '#21252b',
  popoverForeground: '#abb2bf',
  muted: '#2c313a',
  mutedForeground: '#7f848e',
  primary: '#404754',
  primaryForeground: '#abb2bf',
  secondary: '#2c313a',
  secondaryForeground: '#abb2bf',
  accent: '#e06c75',
  accentForeground: '#282c34',
  midground: '#e06c75',
  midgroundForeground: '#282c34',
  border: '#181a1f',
  input: '#1d1f23',
  ring: '#3e4452',
  composerRing: '#3e4452',
  destructive: '#e06c75',
  destructiveForeground: '#282c34',
  sidebarBackground: '#282c34',
  sidebarBorder: '#181a1f',
  userBubble: '#2c313a',
  userBubbleBorder: '#3e4452'
}

const TERMINAL: DesktopTerminalPalette = {
  foreground: '#d7dae0',
  cursor: '#61afef',
  selectionBackground: 'rgba(97, 175, 239, 0.4)',
  black: '#282c34',
  red: '#e06c75',
  green: '#98c379',
  yellow: '#e5c07b',
  blue: '#61afef',
  magenta: '#c678dd',
  cyan: '#56b6c2',
  white: '#abb2bf',
  brightBlack: '#8a919e',
  brightRed: '#ef8b93',
  brightGreen: '#b1d696',
  brightYellow: '#f0d199',
  brightBlue: '#82c2f5',
  brightMagenta: '#d699e8',
  brightCyan: '#74cad5',
  brightWhite: '#e6e6e6'
}

/** Used only for a backend skin, never for an arbitrary installed theme. */
export function oneDarkTheme(theme: DesktopTheme): DesktopTheme {
  const colors = { ...COLORS }
  const terminal = { ...TERMINAL }

  return { ...theme, [ONE_DARK]: true, colors, darkColors: colors, terminal, darkTerminal: terminal } as OneDarkTheme
}

/** Closed renderer policy, not an extensible or persisted CSS-variable bag. */
export function oneDarkProperties(c: DesktopThemeColors): Record<string, string> {
  return {
    '--ui-bg-chrome': c.background,
    '--ui-bg-sidebar': c.sidebarBackground ?? c.background,
    '--ui-bg-editor': c.card,
    '--ui-terminal-surface-background': c.card,
    '--ui-bg-elevated': c.popover,
    '--ui-bg-card': c.card,
    '--ui-bg-input': c.input,
    '--ui-bg-primary': c.primary,
    '--ui-bg-secondary': c.secondary,
    '--ui-bg-tertiary': c.muted,
    '--ui-bg-quaternary': c.muted,
    '--ui-bg-quinary': c.muted,
    '--ui-text-primary': c.foreground,
    '--ui-text-secondary': c.mutedForeground,
    '--ui-text-tertiary': c.mutedForeground,
    '--ui-text-quaternary': c.mutedForeground,
    '--ui-stroke-primary': c.border,
    '--ui-stroke-secondary': c.border,
    '--ui-stroke-tertiary': c.border,
    '--ui-stroke-quaternary': c.border,
    '--ui-row-hover-background': c.muted,
    '--ui-row-active-background': c.muted,
    '--ui-row-open-background': c.muted,
    '--ui-control-hover-background': c.muted,
    '--ui-control-active-background': c.muted,
    '--ui-chat-bubble-background': c.userBubble ?? c.muted,
    '--ui-chat-bubble-opaque-background': c.userBubble ?? c.muted,
    '--ui-selection-background': c.muted,
    '--ui-success': '#98c379',
    '--ui-warning': '#d19a66',
    '--dt-card-foreground': c.cardForeground,
    '--dt-popover': c.popover,
    '--dt-popover-foreground': c.popoverForeground,
    '--dt-muted-foreground': c.mutedForeground,
    // Desktop uses --dt-input for strokes as well as checkbox borders.
    '--dt-input': c.border,
    // The active pane tab's 2px underline follows the accent, not the
    // neutral primary, so it lines up with the sidebar seam's highlight.
    '--pane-tab-active-accent': c.accent,
    '--dt-primary-solid': c.primary,
    '--dt-primary-solid-foreground': c.primaryForeground
  }
}
