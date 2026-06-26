import { type RefObject, useEffect } from 'react'

import { $petInfo } from '@/store/pet'
import { nextScaleFromWheel, PET_SCALE_DEFAULT } from '@/store/pet-gallery'

/** Where the gesture happened + how much it scaled — lets callers zoom toward the
 *  cursor (keep the pixel under it fixed) instead of growing from a corner. */
export interface PetZoomAnchor {
  clientX: number
  clientY: number
  /** new / old scale, clamp-aware (1 at a bound, so the pet doesn't drift). */
  ratio: number
}

/**
 * Wire Alt/Option+wheel-to-scale onto a pet element: hold Alt and scroll up to
 * grow the pet, down to shrink — identical on Mac and Windows. The modifier is
 * required so a plain scroll over the pet still scrolls the page underneath
 * (we only `preventDefault` while Alt is held).
 *
 * Native + non-passive so the conditional `preventDefault` actually takes (and
 * keeps Electron from treating the gesture as a page zoom). Scale is read live
 * from `$petInfo` so rapid steps compound without re-binding the listener.
 *
 * `ready` must track whether the pet element is actually rendered (both callers
 * return null until a pet is enabled). It's a dependency so the listener
 * (re)binds the moment the element mounts — `ref.current` changing alone would
 * never re-run the effect.
 */
export function usePetZoomGesture<T extends HTMLElement>(
  ref: RefObject<T | null>,
  onScale: (scale: number, anchor: PetZoomAnchor) => void,
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

      const base = $petInfo.get().scale ?? PET_SCALE_DEFAULT
      const next = nextScaleFromWheel(base, event.deltaY)
      onScale(next, { clientX: event.clientX, clientY: event.clientY, ratio: next / base })
    }

    el.addEventListener('wheel', onWheel, { passive: false })

    return () => el.removeEventListener('wheel', onWheel)
  }, [ref, onScale, ready])
}
