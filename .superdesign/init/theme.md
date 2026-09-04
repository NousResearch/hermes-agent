# Theme Context

## Compact token summary

- Framework: Tailwind CSS v4 plus @nous-research/ui semantic tokens.
- Default canvas: #041c1c; default text/accent: #ffe6cb.
- Theme layers: background, midground, foreground, each emitted as base/alpha/CSS-mix variables.
- Typography: theme-controlled sans, mono, display, base size, line height, and letter spacing.
- Layout: theme-controlled radius and compact/comfortable/spacious density.
- Components consume shadcn-compatible semantic tokens for card, primary, secondary, muted, accent, border, input, ring, popover, status, and data series.
- Responsive breakpoint used by the shell: 768px.
- Custom user themes may supply exact color overrides, terminal colors, component-style variables, and up to 32 KiB of custom CSS.

## Raw web/src/index.css

```css
@import 'tailwindcss';
/* `fonts.css` must come BEFORE `globals.css`: as of @nous-research/ui 0.14.x,
   `globals.css` only declares the `--font-*` CSS variables (Collapse, Rules
   Compressed/Expanded, Mondwest). The `@font-face` registrations live in
   `fonts.css`, so without this import the DS variables resolve to font
   families the browser never loads and components fall back to a system
   stack (Tabs, Segmented, Typography, Buttons, etc. all look unstyled). */
@import '@nous-research/ui/styles/fonts.css';
@import '@nous-research/ui/styles/globals.css';

/* Scan the published design-system bundle so its utility classes survive
   Tailwind's JIT purge. */
@source '../node_modules/@nous-research/ui/dist';

/* ------------------------------------------------------------------ */
/* JetBrains Mono — bundled for the embedded TUI (/chat tab).          */
/* Gives the terminal a proper monospace font even on systems where    */
/* the user doesn't have one installed locally; xterm.js picks it up   */
/* via ChatPage's `fontFamily` option.                                 */
/* Apache-2.0.                                                         */
/* ------------------------------------------------------------------ */

@font-face {
  font-family: 'JetBrains Mono';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('/fonts-terminal/JetBrainsMono-Regular.woff2') format('woff2');
}
@font-face {
  font-family: 'JetBrains Mono';
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url('/fonts-terminal/JetBrainsMono-Bold.woff2') format('woff2');
}
@font-face {
  font-family: 'JetBrains Mono';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url('/fonts-terminal/JetBrainsMono-Italic.woff2') format('woff2');
}

/* ------------------------------------------------------------------ */
/* Hermes Agent — Nous DS with the LENS_0 (Hermes teal) palette applied
   statically as the default dashboard theme. */
/* ------------------------------------------------------------------ */

:root {
  /* LENS_0 — from design-language/src/ui/components/overlays/index.tsx.
     These are the defaults for the `default` (Hermes Teal) dashboard theme;
     ThemeProvider rewrites them as inline styles when a user switches themes. */
  --foreground: color-mix(in srgb, #ffffff 0%, transparent);
  --foreground-base: #ffffff;
  --foreground-alpha: 0;
  --midground: color-mix(in srgb, #ffe6cb 100%, transparent);
  --midground-base: #ffe6cb;
  --midground-alpha: 1;
  --background: color-mix(in srgb, #041c1c 100%, transparent);
  --background-base: #041c1c;
  --background-alpha: 1;

  /* Typography tokens — rewritten by ThemeProvider. Defaults match the
     system stack so themes that don't override look native. */
  --theme-font-sans: system-ui, -apple-system, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  --theme-font-mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo,
    Consolas, monospace;
  --theme-font-display: var(--theme-font-sans);
  --theme-base-size: 15px;
  --theme-line-height: 1.55;
  --theme-letter-spacing: 0;

  /* Layout tokens. */
  --radius: 0.5rem;
  --theme-radius: 0.5rem;
  --theme-spacing-mul: 1;
  --theme-density: comfortable;

  /* Data-series accents — consumed by Analytics + Models pages for the
     input-vs-output token visualisations (chart bars, table values,
     legend swatches). Defaults are tuned for the Hermes-teal LENS_0
     look: cream input + emerald-400 output read as warm/cool against
     the dark canvas. Themes override via ThemeProvider, which emits
     these as `--series-input-token` / `--series-output-token`. */
  --series-input-token: #ffe6cb;
  --series-output-token: #34d399;
}

/* Theme tokens cascade into the document root so every descendant inherits
   the font stack, base size, and letter spacing without explicit calls. */
html {
  font-family: var(--theme-font-sans);
  font-size: var(--theme-base-size);
  line-height: var(--theme-line-height);
  letter-spacing: var(--theme-letter-spacing);
  height: 100dvh;
  max-height: 100dvh;
  overflow: hidden;
}

body {
  font-family: var(--theme-font-sans);
  min-height: 0;
  height: 100%;
  margin: 0;
  overflow: hidden;
}

code, kbd, pre, samp, .font-mono, .font-mono-ui {
  font-family: var(--theme-font-mono);
}

/* Density: scale the shadcn spacing utilities via a multiplier. The DS
   components use `p-N` / `gap-N` / `space-*` classes which resolve against
   Tailwind's spacing scale; multiplying `--spacing` at :root scales them
   all proportionally in Tailwind v4. */
@theme inline {
  --spacing: calc(0.25rem * var(--theme-spacing-mul, 1));
  --font-sans: var(--theme-font-sans);
  --font-mono: var(--theme-font-mono);
}

#root {
  min-height: 0;
  height: 100%;
  max-height: 100%;
  overflow: hidden;
}

@media (max-width: 768px) {
  html,
  body,
  #root {
    min-height: 100dvh;
    height: auto;
    max-height: none;
    overflow-x: hidden;
    overflow-y: auto;
  }
}

/* Nousnet's hermes-agent layout bumps `small` and `code` to readable
   dashboard sizes. Keep in sync. */
small { font-size: 1.0625rem; }
code { font-size: 0.875rem; }

/* Shadcn-compat tokens.
   The dashboard's page code predates the Nous DS and uses shadcn-style
   utility classes (bg-card, text-muted-foreground, border-border, etc.)
   extensively. Rather than rewrite every call site, we expose those
   tokens on top of the Nous palette so classes continue to resolve. */
@theme inline {
  /* Remap foreground to midground so `text-foreground` / `bg-foreground`
     stay visible — in LENS_0, `--foreground` itself has alpha 0. */
  --color-foreground: var(--midground);

  --color-card: color-mix(in srgb, var(--midground-base) 4%, var(--background-base));
  --color-card-foreground: var(--midground);
  --color-primary: var(--midground);
  --color-primary-foreground: var(--background-base);
  --color-secondary: color-mix(in srgb, var(--midground-base) 6%, var(--background-base));
  --color-secondary-foreground: var(--midground);
  --color-muted: color-mix(in srgb, var(--midground-base) 8%, var(--background-base));
  /* Routes the shadcn `muted-foreground` slot through the DS semantic
     text-secondary token (defaults to midground 80%) so legacy call
     sites that use `text-muted-foreground` get a readable color
     instead of the old 55%-transparent default. */
  --color-muted-foreground: var(--color-text-secondary);
  --color-accent: color-mix(in srgb, var(--midground-base) 10%, var(--background-base));
  --color-accent-foreground: var(--midground);
  --color-destructive: #fb2c36;
  --color-destructive-foreground: #ffffff;
  --color-success: #4ade80;
  --color-warning: #ffbd38;
  --color-border: color-mix(in srgb, var(--midground-base) 15%, transparent);
  --color-input: color-mix(in srgb, var(--midground-base) 15%, transparent);
  --color-ring: var(--midground);
  --color-popover: color-mix(in srgb, var(--midground-base) 4%, var(--background-base));
  --color-popover-foreground: var(--midground);

  --radius-sm: calc(var(--theme-radius) - 4px);
  --radius-md: calc(var(--theme-radius) - 2px);
  --radius-lg: var(--theme-radius);
  --radius-xl: calc(var(--theme-radius) + 4px);
}


/* Collapsed sidebar tooltip entrance — skipped when moving between items. */
@keyframes sidebar-tooltip-in {
  from { opacity: 0; transform: translateY(-50%) translateX(-4px); }
  to   { opacity: 1; transform: translateY(-50%) translateX(0); }
}

/* Toast animations used by `components/Toast.tsx`. */
@keyframes toast-in {
  from { opacity: 0; transform: translateX(16px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes toast-out {
  from { opacity: 1; transform: translateX(0); }
  to   { opacity: 0; transform: translateX(16px); }
}

/* Generic fade + dialog entrance used by popovers and confirm dialogs. */
@keyframes fade-in {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes dialog-in {
  from { opacity: 0; transform: translateY(4px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* Hide scrollbar utility — used by the header's overflow-x nav row. */
.scrollbar-none {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-none::-webkit-scrollbar {
  display: none;
}

/* System UI-monospace stack — distinct from `font-courier` (Courier
   Prime), used for dense data readouts where the display font would
   break the grid. Routes through the theme's mono stack so themes
   with a different monospace (JetBrains Mono, IBM Plex Mono, etc.)
   still apply here. */
.font-mono-ui {
  font-family: var(--theme-font-mono);
}

/* Subtle grain overlay for badges. */
.grain {
  position: relative;
}
.grain::after {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.12;
  pointer-events: none;
  background: repeating-conic-gradient(currentColor 0% 25%, #0000 0% 50%) 0 0 /
    2px 2px;
}

/* RTL support — Arabic and any future right-to-left locale. The i18n provider
   sets `dir` on <html>; Tailwind v4's logical spacing utilities (ms-/me-,
   ps-/pe-) and logical properties then flip automatically. Scoped so the
   default LTR layout is untouched. */
html[dir="rtl"] {
  direction: rtl;
}


```

## Raw web/src/themes/presets.ts

```tsx
import type { DashboardTheme, ThemeTypography, ThemeLayout } from "./types";

/**
 * Built-in dashboard themes.
 *
 * Each theme defines its own palette, typography, and layout so switching
 * themes produces visible changes beyond just color — fonts, density, and
 * corner-radius all shift to match the theme's personality.
 *
 * Theme names must stay in sync with the backend's
 * `_BUILTIN_DASHBOARD_THEMES` list in `hermes_cli/web_server.py`.
 */

// ---------------------------------------------------------------------------
// Shared typography / layout presets
// ---------------------------------------------------------------------------

/** Default system stack — neutral, safe fallback for every platform. */
const SYSTEM_SANS =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const SYSTEM_MONO =
  'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace';

const DEFAULT_TYPOGRAPHY: ThemeTypography = {
  fontSans: SYSTEM_SANS,
  fontMono: SYSTEM_MONO,
  baseSize: "15px",
  lineHeight: "1.55",
  letterSpacing: "0",
};

const DEFAULT_LAYOUT: ThemeLayout = {
  radius: "0.5rem",
  density: "comfortable",
};

// ---------------------------------------------------------------------------
// Themes
// ---------------------------------------------------------------------------

export const defaultTheme: DashboardTheme = {
  name: "default",
  label: "Hermes Teal",
  description: "Classic dark teal — the canonical Hermes look",
  palette: {
    background: { hex: "#041c1c", alpha: 1 },
    midground: { hex: "#ffe6cb", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 189, 56, 0.35)",
    noiseOpacity: 1,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#000000",
};

export const midnightTheme: DashboardTheme = {
  name: "midnight",
  label: "Midnight",
  description: "Deep blue-violet with cool accents",
  palette: {
    background: { hex: "#0a0a1f", alpha: 1 },
    midground: { hex: "#d4c8ff", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(167, 139, 250, 0.32)",
    noiseOpacity: 0.8,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
    letterSpacing: "-0.005em",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.75rem",
  },
};

export const emberTheme: DashboardTheme = {
  name: "ember",
  label: "Ember",
  description: "Warm crimson and bronze — forge vibes",
  palette: {
    background: { hex: "#1a0a06", alpha: 1 },
    midground: { hex: "#ffd8b0", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 115, 22, 0.38)",
    noiseOpacity: 1,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Spectral", Georgia, "Times New Roman", serif`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.25rem",
  },
  colorOverrides: {
    destructive: "#c92d0f",
    warning: "#f97316",
  },
};

export const monoTheme: DashboardTheme = {
  name: "mono",
  label: "Mono",
  description: "Clean grayscale — minimal and focused",
  palette: {
    background: { hex: "#0e0e0e", alpha: 1 },
    midground: { hex: "#eaeaea", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 255, 255, 0.1)",
    noiseOpacity: 0.6,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"IBM Plex Sans", ${SYSTEM_SANS}`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
};

export const cyberpunkTheme: DashboardTheme = {
  name: "cyberpunk",
  label: "Cyberpunk",
  description: "Neon green on black — matrix terminal",
  palette: {
    background: { hex: "#040608", alpha: 1 },
    midground: { hex: "#9bffcf", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(0, 255, 136, 0.22)",
    noiseOpacity: 1.2,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontMono: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
  colorOverrides: {
    success: "#00ff88",
    warning: "#ffd700",
    destructive: "#ff0055",
  },
};

export const roseTheme: DashboardTheme = {
  name: "rose",
  label: "Rosé",
  description: "Soft pink and warm ivory — easy on the eyes",
  palette: {
    background: { hex: "#1a0f15", alpha: 1 },
    midground: { hex: "#ffd4e1", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 168, 212, 0.3)",
    noiseOpacity: 0.9,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Fraunces", Georgia, serif`,
    fontMono: `"DM Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=DM+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "1rem",
  },
};

/** Light mode — vivid Nous-blue accents on a cream canvas. */
export const nousBlueTheme: DashboardTheme = {
  name: "nous-blue",
  label: "Nous Blue",
  description: "Light mode — vivid Nous-blue accents on cream canvas",
  palette: {
    background: { hex: "#E8F2FD", alpha: 1 },
    midground: { hex: "#0053FD", alpha: 1 },
    foreground: { hex: "#170d02", alpha: 0 },
    warmGlow: "rgba(0, 83, 253, 0.12)",
    noiseOpacity: 0,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#f5f8fc",
  terminalForeground: "#170d02",
  seriesColors: {
    inputTokenAccent: "#001934",
    outputTokenAccent: "#0053fd",
  },
  swatchColors: ["#170d02", "#0053FD", "#E8F2FD"],
};

/**
 * Same look as ``defaultTheme`` but with a larger root font size, looser
 * line-height, and ``spacious`` density so every rem-based size in the
 * dashboard scales up. For users who find the default 15px UI too dense.
 */
export const defaultLargeTheme: DashboardTheme = {
  name: "default-large",
  label: "Hermes Teal (Large)",
  description: "Hermes Teal with bigger fonts and roomier spacing",
  palette: defaultTheme.palette,
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    baseSize: "18px",
    lineHeight: "1.65",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    density: "spacious",
  },
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  "nous-blue": nousBlueTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
};

```

## Raw web/src/themes/types.ts

```tsx
/**
 * Dashboard theme model.
 *
 * Themes customise three orthogonal layers:
 *
 *   1. `palette`       — the 3-layer color triplet (background/midground/
 *                         foreground). Legacy `warmGlow` / `noiseOpacity`
 *                         fields remain for theme YAML compat but are unused
 *                         by the lightweight shell.
 *   2. `typography`    — font families, base font size, line height,
 *                         letter spacing. An optional `fontUrl` is injected
 *                         as `<link rel="stylesheet">` so self-hosted and
 *                         Google/Bunny/etc-hosted fonts both work.
 *   3. `layout`        — corner radius and density (spacing multiplier).
 *
 * Plus an optional `colorOverrides` escape hatch for themes that want to
 * pin specific shadcn tokens to exact values (e.g. a pastel theme that
 * needs a softer `destructive` red than the derived default).
 */

/** A color layer: hex base + alpha (0–1). */
export interface ThemeLayer {
  alpha: number;
  hex: string;
}

export interface ThemePalette {
  /** Deepest canvas color (typically near-black). */
  background: ThemeLayer;
  /** Primary text + accent. Most UI chrome reads this. */
  midground: ThemeLayer;
  /** Top-layer highlight. In LENS_0 this is white @ alpha 0 — invisible by
   *  default but still drives `--color-ring`-style accents. */
  foreground: ThemeLayer;
  /** Legacy palette field — kept for theme YAML compat. */
  warmGlow: string;
  /** Legacy palette field — kept for theme YAML compat. */
  noiseOpacity: number;
}

export interface ThemeTypography {
  /** CSS font-family stack for sans-serif body copy. */
  fontSans: string;
  /** CSS font-family stack for monospace / code blocks. */
  fontMono: string;
  /** Optional display/heading font stack. Falls back to `fontSans`. */
  fontDisplay?: string;
  /** Optional external stylesheet URL (e.g. Google Fonts, Bunny Fonts,
   *  self-hosted .woff2 @font-face sheet). Injected as a <link> in <head>
   *  on theme switch. Same URL is never injected twice. */
  fontUrl?: string;
  /** Root font size (controls rem scale). Example: `"14px"`, `"16px"`. */
  baseSize: string;
  /** Default line-height. Example: `"1.5"`, `"1.65"`. */
  lineHeight: string;
  /** Default letter-spacing. Example: `"0"`, `"0.01em"`, `"-0.01em"`. */
  letterSpacing: string;
}

export type ThemeDensity = "compact" | "comfortable" | "spacious";

export interface ThemeLayout {
  /** Corner-radius token. Example: `"0"`, `"0.25rem"`, `"0.5rem"`,
   *  `"1rem"`. Maps to `--radius` and cascades into every component. */
  radius: string;
  /** Spacing multiplier. `compact` = 0.85, `comfortable` = 1.0 (default),
   *  `spacious` = 1.2. Applied via the `--spacing-mul` CSS var. */
  density: ThemeDensity;
}

/** Overall layout variant the shell renders. `standard` = default single-
 *  column page layout. `cockpit` = reserves a left sidebar rail for a
 *  plugin slot (intended for HUD-style themes with persistent status panels).
 *  `tiled` = relaxes the main content max-width so pages can use the full
 *  viewport width. Themes set this; plugins react via CSS vars /
 *  `[data-layout-variant="..."]` selectors. */
export type ThemeLayoutVariant = "standard" | "cockpit" | "tiled";

/** Named hero/background assets a theme can populate. Each value is
 *  emitted as a CSS var (`--theme-asset-<name>`). Plugin slots and
 *  shell chrome may consume these via CSS. */
export interface ThemeAssets {
  /** Full-viewport background image URL. Exposed as `--theme-asset-bg` for
   *  the `backdrop` plugin slot or theme `customCSS`. */
  bg?: string;
  /** Hero render (Gundam, mascot, wallpaper) — for plugin sidebars/overlays. */
  hero?: string;
  /** Logo mark — header slot consumers use this. */
  logo?: string;
  /** Faction/brand crest — header-left decoration. */
  crest?: string;
  /** Secondary sidebar illustration. */
  sidebar?: string;
  /** Alternate header artwork. */
  header?: string;
  /** User-defined named assets. Keyed by [a-zA-Z0-9_-] only.
   *  Emitted as `--theme-asset-custom-<key>`. */
  custom?: Record<string, string>;
}

/** Component-style override buckets. Each bucket's entries become CSS
 *  vars (`--component-<bucket>-<kebab-property>`) that shell components
 *  (Card, App header/footer, etc.) read. Values are plain CSS
 *  strings — we don't parse them, so themes can use `clip-path`,
 *  `border-image`, `background`, `box-shadow`, and anything else CSS
 *  accepts. */
export interface ThemeComponentStyles {
  card?: Record<string, string>;
  header?: Record<string, string>;
  footer?: Record<string, string>;
  sidebar?: Record<string, string>;
  tab?: Record<string, string>;
  progress?: Record<string, string>;
  badge?: Record<string, string>;
  backdrop?: Record<string, string>;
  page?: Record<string, string>;
}

/** Data-series accent colors for chart + table visualisations (Analytics,
 *  Models, etc.). Themes provide hex strings; the provider emits them as
 *  `--series-input-token` / `--series-output-token` CSS vars consumed
 *  inline by pages that render input-vs-output token flows. Themes can
 *  omit either field to inherit the default token defined in
 *  `index.css` (Hermes-teal `#ffe6cb` for input, `#34d399` for output). */
export interface ThemeSeriesColors {
  /** Input-tokens series accent (Analytics chart bars + table values). */
  inputTokenAccent?: string;
  /** Output-tokens series accent. */
  outputTokenAccent?: string;
}

/** Optional hex overrides keyed by shadcn-compat token name (without the
 *  `--color-` prefix). Any key set here wins over the DS cascade. */
export interface ThemeColorOverrides {
  card?: string;
  cardForeground?: string;
  popover?: string;
  popoverForeground?: string;
  primary?: string;
  primaryForeground?: string;
  secondary?: string;
  secondaryForeground?: string;
  muted?: string;
  mutedForeground?: string;
  accent?: string;
  accentForeground?: string;
  destructive?: string;
  destructiveForeground?: string;
  success?: string;
  warning?: string;
  border?: string;
  input?: string;
  ring?: string;
}

export interface DashboardTheme {
  description: string;
  label: string;
  name: string;
  palette: ThemePalette;
  typography: ThemeTypography;
  layout: ThemeLayout;
  /** Overall shell layout. Defaults to `"standard"` when absent. */
  layoutVariant?: ThemeLayoutVariant;
  /** Named + custom asset URLs exposed as CSS vars on theme apply. */
  assets?: ThemeAssets;
  /** Raw CSS injected as a scoped `<style>` tag on theme apply, cleaned up
   *  on theme switch. Intended for selector-level chrome that's too
   *  expressive for componentStyles alone (e.g. `::before` pseudo-elements,
   *  complex animations, media queries). */
  customCSS?: string;
  /** Per-component CSS-var overrides. See `ThemeComponentStyles`. */
  componentStyles?: ThemeComponentStyles;
  colorOverrides?: ThemeColorOverrides;
  /** Data-series accent colors for Analytics/Models token charts. */
  seriesColors?: ThemeSeriesColors;
  /** Explicit 3-color swatch override for the theme picker. Order matches the
   *  default swatch cells: [background, midground, warmGlow]. */
  swatchColors?: [string, string, string];
  /** Background color for the embedded terminal pane (xterm.js).
   *  Hex string. Defaults to `"#000000"` when absent. */
  terminalBackground?: string;
  /** Default text/cursor color for the embedded terminal pane (xterm.js).
   *  Hex string. Defaults to `"#f0e6d2"` when absent. */
  terminalForeground?: string;
}

/**
 * Wire response shape for `GET /api/dashboard/themes`.
 *
 * The `themes` list is intentionally partial — built-in themes are fully
 * defined in `presets.ts`; user themes carry their full definition so the
 * client can apply them without a second round-trip.
 */
export interface ThemeListEntry {
  description: string;
  label: string;
  name: string;
  /** Full theme definition. Present for user-defined themes loaded from
   *  `~/.hermes/dashboard-themes/*.yaml`; undefined for built-ins (the
   *  client already has those in `BUILTIN_THEMES`). */
  definition?: DashboardTheme;
}

export interface ThemeListResponse {
  active: string;
  themes: ThemeListEntry[];
}

```

