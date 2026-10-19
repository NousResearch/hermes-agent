import { atom } from 'nanostores'

import { matchesQuery } from '@/hooks/use-media-query'
import { isBrowserHostedDesktop } from '@/lib/platform'

export const TOUCH_POINTER_QUERY = '(hover: none), (pointer: coarse)'

// One listener shared by every transcript bubble instead of a media-query hook
// per bubble. Guarded like `$narrowViewport`: jsdom and node test envs have no
// `matchMedia`.
const touchPointerQuery = typeof window !== 'undefined' ? window.matchMedia?.(TOUCH_POINTER_QUERY) : undefined

/** Coarse-pointer / no-hover surface, the same query the touch CSS keys on. */
export const $touchPointer = atom(Boolean(touchPointerQuery?.matches))

touchPointerQuery?.addEventListener('change', event => $touchPointer.set(event.matches))

const BROWSER_OWNED_TOUCH_TARGET =
  '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-content"], [data-slot="aui_system-message-root"], [data-selectable-text="true"], input, textarea, [contenteditable]:not([contenteditable="false"]), a[href], img, video, audio'

/** A browser-host touch on text, editables, links or media: its long-press
 * belongs to the browser/OS (selection, paste, link and media callouts),
 * including the interval before selection handles exist. */
export function isBrowserOwnedTouch(event: Event): boolean {
  const element = event.target instanceof Element ? event.target : null

  return Boolean(isBrowserHostedDesktop() && element?.closest(BROWSER_OWNED_TOUCH_TARGET) && isTouchInteraction(event))
}

/** Context-menu/double-click events are not PointerEvents in every browser.
 * Prefer actual input evidence, then WebKit's coarse-pointer capability. */
export function isTouchInteraction(event?: Event): boolean {
  const input = event as
    (Event & { pointerType?: string; sourceCapabilities?: { firesTouchEvents?: boolean } }) | undefined

  if (input?.pointerType) {
    return input.pointerType === 'touch' || input.pointerType === 'pen'
  }

  if (input?.sourceCapabilities) {
    return input.sourceCapabilities.firesTouchEvents === true
  }

  return matchesQuery(TOUCH_POINTER_QUERY)
}
