// Hermes → Boardstate theme mapping (pure, DOM-free — unit-tested in CI).
//
// Hermes themes are whole-palette swaps: each sets shadcn-style `--color-*`
// tokens plus a 3-layer `--background/--midground/--foreground` scale on <html>.
// There is no light/dark toggle and no SDK theme event. We make the board look
// native by aliasing Boardstate's semantic `--bs-*` tokens to the matching Hermes
// source token with `var()`; the DOM glue in `index.tsx` writes these aliases onto
// the `<boardstate-view>` element and re-evaluates the light/dark base whenever the
// host mutates its root tokens.

// Each Boardstate token maps to an ordered list of Hermes source tokens (first
// present wins). Only tokens that carry palette color are mapped — radii, fonts,
// durations and shadows keep the bundle's own values.
export const BS_TO_HERMES: Record<string, string[]> = {
  "--bs-bg": ["--background-base", "--color-background"],
  "--bs-bg-hover": ["--color-secondary"],
  "--bs-bg-muted": ["--color-muted"],
  "--bs-surface-muted": ["--color-muted"],
  "--bs-card": ["--color-card"],
  "--bs-card-highlight": ["--color-secondary"],
  "--bs-border": ["--color-border"],
  "--bs-border-strong": ["--color-input"],
  "--bs-input": ["--color-input"],
  "--bs-text": ["--color-card-foreground", "--foreground-base"],
  "--bs-text-strong": ["--color-primary", "--foreground-base"],
  "--bs-text-muted": ["--color-muted-foreground"],
  "--bs-text-dim": ["--color-text-secondary", "--color-muted-foreground"],
  "--bs-muted": ["--color-muted-foreground"],
  "--bs-accent": ["--color-primary", "--color-accent"],
  "--bs-accent-foreground": ["--color-primary-foreground"],
  "--bs-ring": ["--color-ring"],
  "--bs-danger": ["--color-destructive"],
  "--bs-success": ["--color-success"],
  "--bs-warning": ["--color-warning"],
};

// Build `var(--a, var(--b))` — the innermost candidate has NO fallback, so if
// every Hermes token is absent (older host, non-Hermes embed) the whole
// declaration is invalid and the inline override drops, revealing the bundle's
// own `data-theme` default. This is why a missing host is safe.
//
// The Hermes DESKTOP app (Electron renderer) uses a different token system than the
// web dashboard — `--ui-*` surfaces/text/strokes + `--ui-accent`/`--ui-red`… — so the
// desktop plugin needs its own map. Same var()-alias mechanism; same pure helpers.
//
// ⚠ These are validated against the app's RUNTIME token VALUES, not just names. Trap:
// `--ui-base` is the app's INK color (`var(--theme-foreground)`), NOT a background —
// so page/card surfaces come from `--ui-surface-background` / `--ui-bg-editor` /
// `--ui-bg-elevated`, and the accent's foreground is a light surface, never `--ui-base`.
export const BS_TO_DESKTOP: Record<string, string[]> = {
  "--bs-bg": ["--ui-surface-background", "--ui-bg-editor"],
  "--bs-bg-hover": ["--ui-row-hover-background", "--ui-control-hover-background"],
  "--bs-bg-muted": ["--ui-bg-chrome", "--ui-bg-tertiary"],
  "--bs-surface-muted": ["--ui-bg-chrome", "--ui-bg-tertiary"],
  "--bs-card": ["--ui-bg-elevated", "--ui-surface-background"],
  "--bs-card-highlight": ["--ui-row-active-background", "--ui-bg-elevated"],
  "--bs-border": ["--ui-stroke-secondary"],
  "--bs-border-strong": ["--ui-stroke-primary"],
  "--bs-input": ["--ui-bg-input", "--ui-control-active-background"],
  "--bs-text": ["--ui-text-primary"],
  "--bs-text-strong": ["--ui-text-primary"],
  "--bs-text-muted": ["--ui-text-tertiary"],
  "--bs-text-dim": ["--ui-text-quaternary", "--ui-text-tertiary"],
  "--bs-muted": ["--ui-text-tertiary"],
  "--bs-accent": ["--ui-accent"],
  "--bs-accent-foreground": ["--ui-bg-elevated", "--ui-surface-background"],
  "--bs-ring": ["--ui-accent"],
  "--bs-danger": ["--ui-red"],
  "--bs-success": ["--ui-green"],
  "--bs-warning": ["--ui-yellow"],
};

export function aliasChain(candidates: string[]): string {
  let expr = `var(${candidates[candidates.length - 1]})`;
  for (let i = candidates.length - 2; i >= 0; i--) {
    expr = `var(${candidates[i]}, ${expr})`;
  }
  return expr;
}

// Relative luminance (WCAG) of a CSS color string, 0..1. Handles BOTH `rgb()`/`rgba()`
// (0..255 components) AND the modern `color(srgb r g b)` form that getComputedStyle
// returns for `color-mix()` results (0..1 components) — dividing those by 255 would
// read every light color as near-black and flip the light/dark base the wrong way.
export function relLuminance(color: string): number {
  const m = color.match(/[\d.]+/g);
  if (!m || m.length < 3) return 0;
  const scale = /^\s*color\(/i.test(color) ? 1 : 255;
  const [r, g, b] = m.slice(0, 3).map((n) => {
    const c = Number(n) / scale;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Which built-in palette to fall back to for any un-mapped `--bs-*`, chosen from
// the host's real painted background so the base always matches the active theme.
export function themeBase(backgroundRgb: string): "dark" | "light" {
  return relLuminance(backgroundRgb) < 0.4 ? "dark" : "light";
}
