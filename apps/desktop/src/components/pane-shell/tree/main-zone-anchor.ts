import { atom } from 'nanostores'
import { type RefObject, useCallback, useLayoutEffect } from 'react'

import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'
import { useResizeObserver } from '@/hooks/use-resize-observer'

/** Gap between the window-controls cluster and chrome that must clear it —
 *  the same 12px `usePanelTitlebar` reserves inside a zone's own band. */
const CLUSTER_GAP = 12

/** How far main-view chrome is inset from the zone's own left edge, so the way
 *  back out of a page does not sit flush against the sidebar — the app's chrome
 *  gutter (0.75rem), the same one `usePanelTitlebar` reserves. `panel-titlebar.ts`
 *  shares it: it has to tolerate the inset to still recognise the control as
 *  standing on the zone whose band it reserves. */
export const BACK_CONTROL_INSET = 12

/**
 * Viewport-space left edge (px) where the MAIN zone's band starts, or null until
 * it is measured. Fixed titlebar chrome that belongs to the main VIEW — the way
 * back out of a page — anchors here instead of to the window edge, so it stands
 * at the main view's top-left rather than over the sidebar.
 */
export const $mainZoneLeft = atom<number | null>(null)

/** Publish the anchor. No-op when unchanged, so a busy ResizeObserver never
 *  churns subscribers. */
export function setMainZoneLeft(left: number | null): void {
  if ($mainZoneLeft.get() !== left) {
    $mainZoneLeft.set(left)
  }
}

/**
 * Where the anchor lands: the zone's own left edge, CLAMPED right of the
 * window-controls cluster when the two would overlap. That clamp is the whole
 * point — a sidebar dragged narrower than the cluster (or collapsed to a rail)
 * would otherwise leave main-view chrome riding under the sidebar toggle.
 */
export function mainZoneAnchorLeft(zoneLeft: number, clusterRight: number | null): number {
  // Clamped right of the controls when the zone starts under them (a collapsed
  // or dragged-in sidebar) — no inset there, the sidebar is beneath the cluster.
  // Otherwise the zone's own left edge, inset by the chrome gutter.
  return clusterRight !== null && clusterRight > zoneLeft
    ? Math.round(clusterRight + CLUSTER_GAP)
    : Math.round(zoneLeft + BACK_CONTROL_INSET)
}

/**
 * Measure the main zone and publish its anchor. Enabled for the zone holding the
 * workspace pane — the one a page route renders in — and re-measured whenever
 * that zone, the titlebar clusters, or the window chrome moves.
 */
export function useMainZoneAnchor(ref: RefObject<HTMLElement | null>, enabled: boolean) {
  const measure = useCallback(() => {
    const element = ref.current

    if (!enabled || !element) {
      return
    }

    const rect = element.getBoundingClientRect()

    if (!rect.width) {
      return
    }

    const cluster = document.querySelector<HTMLElement>('[data-titlebar-cluster="left"]')?.getBoundingClientRect()

    setMainZoneLeft(mainZoneAnchorLeft(rect.left, cluster ? cluster.right : null))
  }, [enabled, ref])

  useResizeObserver(measure, ref)

  useLayoutEffect(() => {
    if (!enabled) {
      setMainZoneLeft(null)

      return
    }

    const observer = new ResizeObserver(measure)

    // Re-query on every chrome change: a route switch mounts a different
    // cluster set, and a cluster that is gone has nothing to clamp against.
    const observe = () => {
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

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
      window.removeEventListener(TITLEBAR_CHROME_CHANGED_EVENT, onChromeChanged)
      setMainZoneLeft(null)
    }
  }, [enabled, measure])
}
