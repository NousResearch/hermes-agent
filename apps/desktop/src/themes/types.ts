/**
 * Desktop app theme model.
 *
 *   colors      — Tailwind color tokens written directly to CSS vars.
 *   darkColors  — optional hand-tuned dark variant (else `colors` is reused
 *                 unchanged for dark, and a synth pass generates light).
 *   typography  — font families + optional stylesheet URL.
 *
 * Everything else (layout, sizing, radius, line-height) lives in styles.css.
 * Add new themes in `presets.ts` — no other code changes needed.
 */

export interface DesktopThemeColors {
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
  /**
   * Inline code (`` `like this` ``) foreground / fill / hairline. All optional:
   * when a theme omits them the styles.css `:root` / `:root.dark` defaults win,
   * so existing themes render byte-identically. A theme that wants code tokens
   * to read in its own palette (e.g. lavender identifiers on a green body) sets
   * these and `applyTheme` paints them over the defaults.
   */
  inlineCodeForeground?: string
  inlineCodeBackground?: string
  inlineCodeBorder?: string
  /**
   * Chat markdown headings + bold (`<strong>`). Optional: when omitted the body
   * foreground is used (styles.css `--ui-emphasis-foreground` default), so
   * existing themes are unchanged. A terminal-style skin can set this brighter
   * than `foreground` so bold/headings "brighten" against the body.
   */
  emphasisForeground?: string
  /**
   * Sidebar section headers ("PINNED", "SESSIONS"). Optional: defaults to the
   * brand accent (`--ui-sidebar-heading` → `--theme-primary`). Set e.g. white
   * for a terminal look where headers read as bold white over a tinted sidebar.
   */
  sidebarHeadingForeground?: string
  /**
   * Sidebar entry text (session rows, nav labels). Optional: defaults to the
   * body foreground (`--sidebar-foreground`). Set to decouple sidebar text from
   * chat body text — e.g. near-white rows over a dark-green sidebar.
   */
  sidebarForeground?: string
  /** Top navigation rows in the left sidebar (New session, Skills, Messaging). */
  sidebarNavForeground?: string
  /** Profile/workspace group headings in the left sidebar. */
  sidebarWorkspaceForeground?: string
  /** Child session row text under a profile/workspace group. */
  sidebarSessionForeground?: string
  /** Multiplier for the decorative backdrop image/statue; `0` disables it. */
  backdropOpacity?: string
  /** Optional colored statusbar tones. Defaults preserve the original muted chrome. */
  statusGatewayForeground?: string
  statusAgentsForeground?: string
  statusCronForeground?: string
  statusContextForeground?: string
  statusSessionForeground?: string
  statusYoloForeground?: string
  statusTerminalForeground?: string
  statusVersionForeground?: string
}

export interface DesktopThemeTypography {
  fontSans: string
  fontMono: string
  /** Google/Bunny/self-hosted font stylesheet URL. */
  fontUrl?: string
}

/**
 * Integrated-terminal ANSI palette (xterm `ITheme`, minus `background`).
 *
 * Populated only when a converted VS Code theme ships a full `terminal.ansi*`
 * set; otherwise the terminal keeps its built-in VS Code default palette.
 * `background` is intentionally absent — the pane always paints the live skin
 * surface so it stays translucent.
 */
export interface DesktopTerminalPalette {
  foreground?: string
  cursor?: string
  /** Keeps its source alpha — xterm blends it over the surface. */
  selectionBackground?: string
  black?: string
  red?: string
  green?: string
  yellow?: string
  blue?: string
  magenta?: string
  cyan?: string
  white?: string
  brightBlack?: string
  brightRed?: string
  brightGreen?: string
  brightYellow?: string
  brightBlue?: string
  brightMagenta?: string
  brightCyan?: string
  brightWhite?: string
}

export interface DesktopTheme {
  name: string
  label: string
  description: string
  /** Light palette (also reused for dark when `darkColors` is omitted). */
  colors: DesktopThemeColors
  /** Hand-tuned dark palette. Skins like `nous` ship one. */
  darkColors?: DesktopThemeColors
  typography?: Partial<DesktopThemeTypography>
  /**
   * Shiki syntax-highlighting theme for fenced code blocks, per color mode.
   * Optional — omitted means the app default (`github-dark-default` /
   * `github-light-default`). Each side is a bundled Shiki theme name; a missing
   * side falls back to the default for that mode. Lets a skin ship code blocks
   * that match its surface instead of always GitHub-dark.
   */
  shikiTheme?: { light?: string; dark?: string }
  /** Light-variant terminal ANSI palette (also the fallback for dark). */
  terminal?: DesktopTerminalPalette
  /** Dark-variant terminal ANSI palette. Falls back to `terminal`. */
  darkTerminal?: DesktopTerminalPalette
}
