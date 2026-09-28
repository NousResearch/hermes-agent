// Locks in the compact, voice-first composer geometry (see compact-layout.ts).
// The app is a voice agent first: the text field must stay a discreet one-line
// add-on. If someone re-inflates the field (bigger cap, bigger type, looser
// padding) these invariants go red instead of the layout silently growing back.
import { describe, expect, it } from 'vitest'

import { COMPOSER_COMPACT_FADE, COMPOSER_COMPACT_INPUT, COMPOSER_COMPACT_LAYOUT } from './compact-layout'

// The `[data-slot='composer-rich-input']` font size styles.css ships — the
// compact field must read SMALLER than this.
const STYLES_DEFAULT_INPUT_FONT_REM = 0.8125
// The old `--composer-input-max-height` (:root default) that allowed ~7 lines.
const OLD_INPUT_MAX_HEIGHT_REM = 9.375

/** Read a `[--name:<n>rem]` override out of a Tailwind class string. */
function remVar(source: string, name: string): number {
  const match = source.match(new RegExp(`\\[--${name}:([0-9.]+)rem\\]`))

  expect(match, `${name} override missing from the compact layout`).not.toBeNull()

  return Number(match![1])
}

describe('compact voice-first composer geometry', () => {
  it('caps the field at ~3 lines and keeps one line as the floor', () => {
    const min = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-input-min-height')
    const max = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-input-max-height')
    // The field's line box is 1rem (`leading-4` on the compact input).
    const maxLines = max / 1

    expect(min).toBeLessThanOrEqual(1.25)
    expect(maxLines).toBeGreaterThanOrEqual(2)
    expect(maxLines).toBeLessThanOrEqual(3)
    // And clearly below the old cap.
    expect(max).toBeLessThan(OLD_INPUT_MAX_HEIGHT_REM)
  })

  it('shrinks the field type below the styles.css default and keeps one short line', () => {
    const font = Number(COMPOSER_COMPACT_INPUT.match(/text-\[([0-9.]+)rem\]!/)![1])

    expect(font).toBeLessThan(STYLES_DEFAULT_INPUT_FONT_REM)
    expect(COMPOSER_COMPACT_INPUT).toContain('leading-4')
    expect(COMPOSER_COMPACT_INPUT).toContain('max-h-(--composer-input-max-height)')
    expect(COMPOSER_COMPACT_INPUT).toContain('min-h-(--composer-input-min-height)')
  })

  it('tightens the chrome and keeps the primary control the largest', () => {
    const control = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-control-size')
    const primary = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-control-primary-size')
    const padY = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-surface-pad-y')
    const rowGap = remVar(COMPOSER_COMPACT_LAYOUT, 'composer-row-gap')

    expect(padY).toBeLessThanOrEqual(0.25)
    expect(rowGap).toBeLessThanOrEqual(0.25)
    // Smaller than the styles.css defaults (control 1.5rem, primary 1.625rem).
    expect(control).toBeLessThan(1.5)
    expect(primary).toBeLessThan(1.625)
    // The voice/send circle still leads the row.
    expect(primary).toBeGreaterThanOrEqual(control)
  })

  it('lowers the empty-composer floor to the compact row height', () => {
    expect(COMPOSER_COMPACT_FADE).toContain('min-h-[1.75rem]!')
  })
})
