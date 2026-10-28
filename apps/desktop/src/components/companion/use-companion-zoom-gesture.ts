import { type RefObject, useEffect } from 'react'

import { $companionInfo } from '@/store/companion'
import { COMPANION_SCALE_DEFAULT, nextScaleFromWheel } from '@/store/companion-gallery'

/** Where the gesture happened + how much it scaled — lets callers zoom toward the
 *  cursor (keep the pixel under it fixed) instead of growing from a corner. */
export interface CompanionZoomAnchor {
  clientX: number
  clientY: number
  /** new / old scale, clamp-aware (1 at a bound, so the companion doesn't drift). */
  ratio: number
}

/**
 * Wire Alt/Option+wheel-to-scale onto a companion element: hold Alt and scroll up to
 * grow the companion, down to shrink — identical on Mac and Windows. The modifier is
 * required so a plain scroll over the companion still scrolls the page underneath
 * (we only `preventDefault` while Alt is held).
 *
 * Native + non-passive so the conditional `preventDefault` actually takes (and
 * keeps Electron from treating the gesture as a page zoom). Scale is read live
 * from `$companionInfo` so rapid steps compound without re-binding the listener.
 *
 * `ready` must track whether the companion element is actually rendered (both callers
 * return null until a companion is enabled). It's a dependency so the listener
 * (re)binds the moment the element mounts — `ref.current` changing alone would
 * never re-run the effect.
 */
export function useCompanionZoomGesture<T extends HTMLElement>(
  ref: RefObject<T | null>,
  onScale: (scale: number, anchor: CompanionZoomAnchor) => void,
  ready: boolean
): void {
  useEffect(() => {
    const el = ref.current

    if (!el || !ready) {
      return
    }

    const onWheel = (event: WheelEvent) => {
      if (!event.altKey) {
        return
      }

      event.preventDefault()

      const base = $companionInfo.get().scale ?? COMPANION_SCALE_DEFAULT
      const next = nextScaleFromWheel(base, event.deltaY)
      onScale(next, { clientX: event.clientX, clientY: event.clientY, ratio: next / base })
    }

    el.addEventListener('wheel', onWheel, { passive: false })

    return () => el.removeEventListener('wheel', onWheel)
  }, [ref, onScale, ready])
}
