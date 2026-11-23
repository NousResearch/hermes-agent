import { useLayoutEffect, useRef } from 'react'

import { useRootAttribute } from './root-attribute'

/** The floor's height on <html>, read by the redesign's toast region (redesign/portal-shell.css). */
export const TOAST_FLOOR_HEIGHT = '--toast-floor-height'

/**
 * Marks an action row at the window's floor that toasts must rest above while it is mounted: a
 * sheet's footer sits in the corner toasts rise from, and a toast over it covers Save and Cancel.
 * The footer cannot be the toast region's CSS anchor (a sheet is portalled after the region, and an
 * anchor has to come first), so its height is published on <html> and kept current as it wraps.
 */
export function useToastFloor<T extends HTMLElement>() {
  const ref = useRef<T>(null)

  useRootAttribute('data-toast-floor', true)

  useLayoutEffect(() => {
    const floor = ref.current

    if (!floor) {
      return
    }

    const root = document.documentElement
    const publish = () => root.style.setProperty(TOAST_FLOOR_HEIGHT, `${floor.offsetHeight}px`)
    const observer = new ResizeObserver(publish)

    observer.observe(floor)
    publish()

    return () => {
      observer.disconnect()
      root.style.removeProperty(TOAST_FLOOR_HEIGHT)
    }
  }, [])

  return ref
}
