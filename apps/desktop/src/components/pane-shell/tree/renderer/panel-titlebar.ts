import { type RefObject, useCallback, useLayoutEffect, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'

/** Reserve actual chrome intersections, including after a neighbor becomes a rail. */
export function usePanelTitlebar(ref: RefObject<HTMLElement | null>, enabled: boolean, minimized: boolean) {
  const [belowControls, setBelowControls] = useState(true)

  const measure = useCallback(() => {
    const element = ref.current

    if (!enabled || !element) {
      return
    }

    const rect = element.getBoundingClientRect()

    if (!rect.width) {
      return
    }

    const leftControls = document.querySelector<HTMLElement>('[data-titlebar-cluster="left"]')?.getBoundingClientRect()

    const rightControls = document
      .querySelector<HTMLElement>('[data-titlebar-cluster="right"]')
      ?.getBoundingClientRect()

    // Empty chrome during a route overlay retains the last safe reservation.
    if (!leftControls || !rightControls) {
      return
    }

    const left = Math.min(rect.width, Math.max(0, leftControls.right + 12 - rect.left))
    const right = Math.min(rect.width - left, Math.max(0, rect.right - rightControls.left + 24))
    element.style.setProperty('--panel-titlebar-left', `${left}px`)
    element.style.setProperty('--panel-titlebar-right', `${right}px`)
    setBelowControls(minimized || rect.width - left - right < 120)
  }, [enabled, minimized, ref])

  useResizeObserver(measure, ref)
  useLayoutEffect(() => {
    if (!enabled) {
      return
    }

    measure()
    const observer = new ResizeObserver(measure)

    for (const element of document.querySelectorAll('[data-titlebar-cluster], [data-tree-group]')) {
      observer.observe(element)
    }

    window.addEventListener('resize', measure)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [enabled, measure])

  return enabled && belowControls
}
