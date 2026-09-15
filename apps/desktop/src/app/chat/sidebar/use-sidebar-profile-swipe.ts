import { type RefObject, useEffect } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { createHorizontalSwipeDetector } from '@/lib/trackpad-gestures'
import { cycleProfile } from '@/store/profile'

/** Own horizontal trackpad swipes that begin anywhere in the Sessions sidebar. */
export function useSidebarProfileSwipe(sidebarRef: RefObject<HTMLElement | null>, enabled: boolean): void {
  useEffect(() => {
    const sidebar = sidebarRef.current

    if (!sidebar || !enabled) {
      return
    }

    const detectSwipe = createHorizontalSwipeDetector()

    const onWheel = (event: WheelEvent) => {
      // The rail owns horizontal scrolling. Full native swipes still route
      // through the whole sidebar, including the rail.
      if (event.target instanceof Element && event.target.closest('[data-slot="profile-rail"]')) {
        return
      }

      // Let a nested control keep any wheel gesture it deliberately consumed.
      if (event.defaultPrevented) {
        return
      }

      const swipe = detectSwipe(event)

      if (!swipe.claimed) {
        return
      }

      event.preventDefault()

      if (swipe.direction !== null) {
        cycleProfile(swipe.direction)
        triggerHaptic('selection')
      }
    }

    sidebar.addEventListener('wheel', onWheel, { passive: false })

    return () => sidebar.removeEventListener('wheel', onWheel)
  }, [enabled, sidebarRef])
}
