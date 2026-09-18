/**
 * Message bubble transparency — how much of your own bubble's fill shows.
 *
 * One 0–100 lever, same direction as Window Translucency: 0 keeps the bubble
 * solid (default), 100 leaves only the outline. Presentation-only, so the
 * renderer owns it (desktop AGENTS.md: state lives with its authority) and
 * paints it as ONE root variable — `--user-bubble-keep`, the share of the
 * theme's bubble fill to keep — that `--dt-user-bubble` mixes in (styles.css).
 * The bubble and the inline edit composer share that token, so both follow.
 */

import { clampIntensity, TRANSLUCENCY_MAX, TRANSLUCENCY_MIN } from '@hermes/shared/translucency'
import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.user-bubble-transparency.v1'

export const $userBubbleTransparency = atom<number>(
  typeof window === 'undefined' ? TRANSLUCENCY_MIN : clampIntensity(storedString(KEY))
)

export function setUserBubbleTransparency(value: number): void {
  $userBubbleTransparency.set(clampIntensity(value))
}

if (typeof window !== 'undefined') {
  $userBubbleTransparency.subscribe(value => {
    const root = document.documentElement

    // The default paints nothing: styles.css falls back to the full fill, so a
    // user who never touched the lever gets byte-identical CSS to before.
    if (value === TRANSLUCENCY_MIN) {
      root.style.removeProperty('--user-bubble-keep')
      persistString(KEY, null)
    } else {
      root.style.setProperty('--user-bubble-keep', `${TRANSLUCENCY_MAX - value}%`)
      persistString(KEY, String(value))
    }
  })
}
