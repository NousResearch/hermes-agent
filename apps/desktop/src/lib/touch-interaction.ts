import { matchesQuery } from '@/hooks/use-media-query'

export const TOUCH_POINTER_QUERY = '(hover: none), (pointer: coarse)'

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
