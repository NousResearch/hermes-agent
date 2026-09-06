/**
 * Guided tours (the `tour` tool) — on, with a switch to stop them.
 *
 * A tour takes the screen: it dims the app, spotlights an element, and pages
 * with Next/Prev. That is worth having and worth being able to refuse, and
 * unlike tips there is no ambient half to separate out — every tour is Hermes
 * running one, so the switch governs the whole feature.
 *
 * Renderer-owned because the renderer is what a tour happens to. Turning it off
 * makes the app decline a tour request outright rather than quietly no-op it,
 * so the agent learns the tour didn't run and can say so in words instead.
 */

import { Codecs, persistentAtom } from '@/lib/persisted'
import { mirrorDisplayToggle } from '@/store/display-toggles'

const KEY = 'hermes.desktop.tours.v1'

export const $toursEnabled = persistentAtom(KEY, true, Codecs.bool)

// Off has to reach the agent, not just the renderer: the `tour` tool leaves the
// model's schema entirely rather than staying on offer and failing.
mirrorDisplayToggle('display.in_app_tours', KEY, $toursEnabled)

export function setToursEnabled(enabled: boolean): void {
  $toursEnabled.set(enabled)
}
