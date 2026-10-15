import { type RefObject, useCallback, useLayoutEffect, useState } from 'react'

import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'
import { BACK_CONTROL_INSET } from '@/components/pane-shell/tree/main-zone-anchor'
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

    // The window-controls cluster and the way back out of a page both stand at
    // the LEFT edge of whatever they cover, so the zone's band has to clear
    // both. The back control is not window chrome — it is anchored to the zone a
    // page renders in (main-zone-anchor.ts) — so only the zone it actually
    // stands on reserves it; the other top-edge zones start to its right and
    // must keep their own band.
    const chromeRight = leftControls.right + 12
    const backControls = document.querySelector<HTMLElement>('[data-titlebar-back]')?.getBoundingClientRect()

    // The control is inset from the zone's own left edge (BACK_CONTROL_INSET), so
    // the geometric test tolerates that much before it reads as another zone's.
    const backStandsHere = Boolean(
      backControls && backControls.left <= Math.max(rect.left, chromeRight) + BACK_CONTROL_INSET
    )

    const backRight = backStandsHere && backControls ? backControls.right + 12 : 0

    const left = Math.min(rect.width, Math.max(0, Math.max(chromeRight, backRight) - rect.left))
    // No extra pad on the right: a +N here is a dead strip (the spacer has no
    // app-region) flush against Settings. The 48px data-window-drag-handle is
    // already the gutter (#131729). Do not mark this spacer drag — it sits
    // under the z-70 cluster, and Electron app-region ignores stacking.
    // The same rule is why the left reservation must cover the back control: its
    // band's flexible remainder IS a drag region (the native drag handle, or a
    // titlebar tab strip), and a drag region wins hit-testing over anything
    // painted above it — a button inside it is visible and dead.
    const right = Math.min(rect.width - left, Math.max(0, rect.right - rightControls.left))
    element.style.setProperty('--panel-titlebar-left', `${left}px`)
    element.style.setProperty('--panel-titlebar-right', `${right}px`)
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

    const onChromeChanged = () => {
      observe()
      measure()
    }

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
