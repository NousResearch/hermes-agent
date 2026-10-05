/**
 * Reduce visual effects — the per-app counterpart to the OS accessibility
 * signals (`prefers-reduced-transparency`, `prefers-reduced-motion`).
 *
 * The media queries in `styles.css` already turn the expensive rendering off,
 * but they are MACHINE-wide: asking for them so that one app stops costing
 * frames takes the blur and the motion away from every other app on the
 * desktop. This is the same trade, scoped to Hermes.
 *
 * Applied as an attribute on `<html>` rather than through each consumer,
 * because the cost is not one component's. `backdrop-filter` re-samples its
 * whole region on every frame that region changes (see the reasoning on the
 * reduced-transparency gate in styles.css), and the decorative CSS animations
 * keep producing frames whether or not anyone is looking at them.
 * `styles.css` keys its blanket gates off this attribute exactly as it already
 * keys them off the two media queries — same resets, different trigger.
 *
 * Scope: this stops the compositing/paint cost. It deliberately does NOT
 * change layout, typography, colours or the transcript budgets, so it can't
 * make the app render something different from what the user expects.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

export const REDUCE_EFFECTS_ATTRIBUTE = 'data-reduce-effects'

const KEY = 'hermes.desktop.reduce-effects.v1'

/** Absent key keeps full effects — the pre-toggle default for existing installs. */
export const $reduceEffects = atom<boolean>(typeof window === 'undefined' ? false : storedString(KEY) === 'on')

export function setReduceEffects(enabled: boolean): void {
  $reduceEffects.set(enabled)
}

// Desktop-local presentation, independent of the active profile — the same
// shape as store/chat-text-scale.ts. The subscribe fires immediately with the
// stored value, so the attribute is correct before the first paint.
if (typeof window !== 'undefined') {
  $reduceEffects.subscribe(enabled => {
    document.documentElement.toggleAttribute(REDUCE_EFFECTS_ATTRIBUTE, enabled)
    persistString(KEY, enabled ? 'on' : null)
  })
}
