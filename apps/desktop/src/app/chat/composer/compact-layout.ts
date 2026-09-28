import { cn } from '@/lib/utils'

/**
 * Compact, voice-first composer geometry.
 *
 * Agent Czesiek is a voice agent first: the text field is a discreet add-on
 * beside the mic, never the main surface. Everything here is a LOCAL override
 * applied on the composer ROOT (CSS variables + a couple of utilities), so it
 * cascades to the input and the control row WITHOUT touching the global `:root`
 * defaults in styles.css that the HUD fallback and the inline edit-composer
 * still read.
 *
 * Keep every value literal — Tailwind scans class strings statically, so an
 * interpolated arbitrary value would generate no CSS at all.
 */

/**
 * Composer geometry, as CSS-variable overrides on the composer root. These win
 * over the `:root` defaults because a value set on the element itself beats a
 * value inherited from the root, regardless of cascade layer.
 *
 * The field is one line that grows to at most ~3 before it scrolls; the frame
 * around it is a hairline, not a padded panel; the controls shrink with the
 * bar but the round primary (mic / send) stays the largest thing in the row.
 */
export const COMPOSER_COMPACT_LAYOUT = cn(
  // The field: one line, capped at ~3 lines before it scrolls.
  '[--composer-input-min-height:1.25rem]',
  '[--composer-input-max-height:3rem]',
  // Tighter chrome around the field and the row.
  '[--composer-surface-pad-y:0.125rem]',
  '[--composer-surface-pad-x:0.375rem]',
  '[--composer-row-gap:0.125rem]',
  '[--composer-control-gap:0.1875rem]',
  // Controls shrink with the bar; the primary stays the biggest by a hair.
  '[--composer-control-size:1.375rem]',
  '[--composer-control-primary-size:1.4375rem]'
)

/**
 * The field itself: one short line, growing to the capped height then scrolling.
 *
 * `text-[0.75rem]!` and `leading-4` take the type below the
 * `[data-slot='composer-rich-input']` default (0.8125rem) and tighten the line
 * box to 1rem. The `!` is required because that styles.css rule is UNLAYERED —
 * a plain utility (utilities layer) loses to an unlayered rule, so the font
 * size could not otherwise be brought down from this component.
 */
export const COMPOSER_COMPACT_INPUT = cn(
  'min-h-(--composer-input-min-height) max-h-(--composer-input-max-height)',
  'py-0.5 pr-1 leading-4 text-[0.75rem]!'
)

/**
 * The empty-composer floor, lowered to match the compact row (the controls are
 * 22px + 2px padding + 2px border ≈ 28px). The `!` is required again:
 * `[data-slot='composer-fade'] { min-height: 2.375rem }` is unlayered, so a
 * plain utility could not lower it.
 */
export const COMPOSER_COMPACT_FADE = 'min-h-[1.75rem]!'
