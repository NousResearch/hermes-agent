/**
 * Source of truth for the `--ui-text-*` ramp in styles.css.
 *
 * Each rung paints `--ui-base` at an alpha over the surface beneath (see the
 * `color-mix(in srgb, var(--ui-base) P%, transparent)` definitions in
 * styles.css). Keeping the percentages here - and having text-contrast.test.ts
 * parse styles.css to confirm they match - means the ramp cannot drift between
 * the CSS and the contrast guard: a change to either side fails the test
 * instead of silently shipping stale numbers.
 */
export const TEXT_RAMP = {
  primary: 0.94,
  // `--ui-text-secondary` drives `--dt-secondary-foreground`.
  secondary: 0.74,
  // `--ui-text-tertiary` drives the status bar and command-center captions.
  tertiary: 0.68,
  // `--ui-text-quaternary` drives timestamps and empty-state counts.
  // Intentionally equal to tertiary: at these alphas no gap both keeps a
  // visual step and clears AA, so the finer hierarchy is traded for readable
  // text.
  quaternary: 0.68
} as const

/**
 * The CSS custom property each ramp rung maps to, keyed to match `TEXT_RAMP`.
 * The test parses these exact declarations out of styles.css to prove the
 * source-of-truth values and production are the same thing.
 */
export const TEXT_RAMP_CSS_VARS = {
  primary: '--ui-text-primary',
  secondary: '--ui-text-secondary',
  tertiary: '--ui-text-tertiary',
  quaternary: '--ui-text-quaternary'
} as const
