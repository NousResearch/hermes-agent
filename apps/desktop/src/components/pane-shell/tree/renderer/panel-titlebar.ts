import { type RefObject, useCallback, useLayoutEffect, useState } from 'react'

import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { $connection } from '@/store/session'

/** The window-chrome inputs that move the fixed titlebar clusters without
 *  changing their size (macOS fullscreen hides the traffic lights → the left
 *  cluster pins to the window edge). */
function chromeKey(): string {
  const connection = $connection.get()
  const position = connection?.windowButtonPosition

  return `${connection?.isFullscreen ? 1 : 0}:${position?.x ?? ''}:${position?.y ?? ''}`
}

function setReservation(element: HTMLElement, name: string, value: string) {
  if (element.style.getPropertyValue(name) !== value) {
    element.style.setProperty(name, value)
  }
}

/** Reserve actual chrome intersections, including after a neighbor becomes a rail. */
export function usePanelTitlebar(ref: RefObject<HTMLElement | null>, enabled: boolean, minimized: boolean) {
  const [belowControls, setBelowControls] = useState(true)

  const measure = useCallback(() => {
    const element = ref.current

    if (!enabled || !element) {
      return
    }

    // Clusters first: during a route overlay there are none, and reading the panel would force a
    // layout of whatever the route just mounted for nothing. Empty chrome keeps the last reservation.
    const leftControls = document.querySelector<HTMLElement>('[data-titlebar-cluster="left"]')
    const rightControls = document.querySelector<HTMLElement>('[data-titlebar-cluster="right"]')

    if (!leftControls || !rightControls) {
      return
    }

    const rect = element.getBoundingClientRect()

    if (!rect.width) {
      return
    }

    const leftRect = leftControls.getBoundingClientRect()
    const rightRect = rightControls.getBoundingClientRect()
    const left = Math.min(rect.width, Math.max(0, leftRect.right + 12 - rect.left))
    // No extra pad on the right: a +N here is a dead strip (the spacer has no
    // app-region) flush against Settings. The 48px data-window-drag-handle is
    // already the gutter (#131729). Do not mark this spacer drag — it sits
    // under the z-70 cluster, and Electron app-region ignores stacking.
    const right = Math.min(rect.width - left, Math.max(0, rect.right - rightRect.left))
    // Write only a change: the properties inherit through the whole pane, so an identical write
    // still restyles every descendant and makes the next panel's read lay the page out again.
    setReservation(element, '--panel-titlebar-left', `${left}px`)
    setReservation(element, '--panel-titlebar-right', `${right}px`)
    setBelowControls(minimized || rect.width - left - right < 120)
  }, [enabled, minimized, ref])

  useResizeObserver(measure, ref)
  useLayoutEffect(() => {
    if (!enabled) {
      return
    }

    const observer = new ResizeObserver(measure)

    const observe = () => {
      // Re-query on every chrome change: route switches mount a different
      // cluster set (app clusters vs a page-owned band), and observing the
      // unmounted set would measure nothing.
      observer.disconnect()

      for (const element of document.querySelectorAll('[data-titlebar-cluster], [data-tree-group]')) {
        observer.observe(element)
      }
    }

    // A route switch mounts a different cluster set. Observing it is enough: the observer's first
    // delivery for each new element measures after the frame's layout, where the reads are free,
    // instead of forcing a layout of the page the route just mounted (that cost 20-190ms a click).
    const onChromeChanged = observe

    observe()
    measure()
    window.addEventListener('resize', measure)
    window.addEventListener(TITLEBAR_CHROME_CHANGED_EVENT, onChromeChanged)

    // A fullscreen transition first fires `resize` (measured against the
    // pre-transition cluster) and only then lands the window-state IPC that
    // translates the fixed clusters — same size, new position — so neither
    // ResizeObserver nor `resize` re-runs. Re-measure after the frame that
    // repaints the clusters from the new chrome vars.
    let lastChrome = chromeKey()
    let frame = 0

    const unsubscribeChrome = $connection.subscribe(() => {
      const nextChrome = chromeKey()

      if (nextChrome === lastChrome) {
        return
      }

      lastChrome = nextChrome
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        frame = requestAnimationFrame(measure)
      })
    })

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
      window.removeEventListener(TITLEBAR_CHROME_CHANGED_EVENT, onChromeChanged)
      unsubscribeChrome()
      cancelAnimationFrame(frame)
    }
  }, [enabled, measure])

  return enabled && belowControls
}
