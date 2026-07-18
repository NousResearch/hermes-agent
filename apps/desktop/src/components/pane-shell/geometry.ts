/**
 * Pane geometry — AABB INTERSECTION → WINDOW-CONTROL AWARENESS. The native
 * window controls (macOS traffic lights top-left / Windows-WSLg overlay
 * top-right) are a rectangle in viewport pixels. Any region whose rect
 * intersects it must reserve that space and expose a drag strip. One
 * `intersect()` call replaces per-layout inset special cases.
 *
 * Also publishes the WORKSPACE zone's edges as CSS vars (see
 * publishWorkspaceGeometry) — chrome that aligns to the main pane (titlebar
 * title et al) consumes `var(--workspace-left/right)` in plain CSS instead of
 * threading rects through React.
 */

import { type RefObject, useLayoutEffect, useState, useSyncExternalStore } from 'react'

import { $layoutTree } from '@/components/pane-shell/tree/store'
import { $connection } from '@/store/session'

// ---------------------------------------------------------------------------
// Rects
// ---------------------------------------------------------------------------

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/** AABB intersection. Returns null when the rects don't overlap. */
export function intersect(a: Rect, b: Rect): Rect | null {
  const x = Math.max(a.x, b.x)
  const y = Math.max(a.y, b.y)
  const right = Math.min(a.x + a.width, b.x + b.width)
  const bottom = Math.min(a.y + a.height, b.y + b.height)

  if (right <= x || bottom <= y) {
    return null
  }

  return { x, y, width: right - x, height: bottom - y }
}

// ---------------------------------------------------------------------------
// Native window controls rect
// ---------------------------------------------------------------------------

/** Height of the band the native controls live in. */
const CONTROLS_BAND_HEIGHT = 34
/** Width of the macOS traffic-light cluster measured from the buttons' x. */
const MACOS_LIGHTS_WIDTH = 58
const MACOS_FALLBACK_BUTTON_X = 24

interface ConnectionLike {
  windowButtonPosition?: { x: number; y: number } | null
  nativeOverlayWidth?: number | null
  isFullscreen?: boolean | null
}

/**
 * The native window-control rectangle in viewport pixels, or null when there
 * is nothing to dodge (fullscreen, plain browser, secondary windows with
 * hidden controls).
 */
export function windowControlsRect(connection: ConnectionLike | null, viewportWidth: number): Rect | null {
  const inElectron = typeof window !== 'undefined' && 'hermesDesktop' in window

  if (!inElectron) {
    return null
  }

  if (connection?.isFullscreen) {
    return null
  }

  // Windows / WSLg: native overlay on the top-right.
  const overlayWidth = connection?.nativeOverlayWidth ?? 0

  if (overlayWidth > 0) {
    return { x: viewportWidth - overlayWidth, y: 0, width: overlayWidth, height: CONTROLS_BAND_HEIGHT }
  }

  // macOS: traffic lights on the top-left. windowButtonPosition === null means
  // the platform has no left-side controls at all (Windows/Linux w/o overlay).
  const pos = connection?.windowButtonPosition

  if (pos === null) {
    return null
  }

  const isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform)

  if (!pos && !isMac) {
    return null
  }

  const x = pos?.x ?? MACOS_FALLBACK_BUTTON_X

  return { x: 0, y: 0, width: x + MACOS_LIGHTS_WIDTH, height: CONTROLS_BAND_HEIGHT }
}

// ---------------------------------------------------------------------------
// Live hook
// ---------------------------------------------------------------------------

let cachedRect: Rect | null = null
let cachedKey = ''

function rectKey(r: Rect | null) {
  return r ? `${r.x},${r.y},${r.width},${r.height}` : ''
}

function readControlsRect(): Rect | null {
  const next = windowControlsRect($connection.get(), typeof window === 'undefined' ? 0 : window.innerWidth)
  const key = rectKey(next)

  // Referentially stable snapshot for useSyncExternalStore.
  if (key !== cachedKey) {
    cachedKey = key
    cachedRect = next
  }

  return cachedRect
}

function subscribeControlsRect(cb: () => void) {
  const unsubConnection = $connection.subscribe(() => cb())
  window.addEventListener('resize', cb)

  return () => {
    unsubConnection()
    window.removeEventListener('resize', cb)
  }
}

/** Reactive native window-controls rect (connection + viewport aware). */
export function useWindowControlsRect(): Rect | null {
  return useSyncExternalStore(subscribeControlsRect, readControlsRect, () => null)
}

// ---------------------------------------------------------------------------
// Per-element overlap
// ---------------------------------------------------------------------------

function sameRect(a: Rect | null, b: Rect | null) {
  if (a === b) {
    return true
  }

  if (!a || !b) {
    return false
  }

  return a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height
}

// ---------------------------------------------------------------------------
// Workspace-edge CSS vars
// ---------------------------------------------------------------------------

/**
 * Publish the workspace zone's viewport edges as root CSS vars:
 *   --workspace-left  : px from the viewport's left to the main zone
 *   --workspace-right : px from the main zone's right to the viewport's right
 *
 * Measured once per relevant change (zone resize via ResizeObserver, layout
 * mutations via $layoutTree, window resize) and consumed in plain CSS — the
 * measured-var idiom (--composer-measured-height), not per-component JS
 * geometry. Call once from the tree root; returns the disposer.
 */
export function publishWorkspaceGeometry(): () => void {
  const root = document.documentElement
  let el: HTMLElement | null = null
  let lastLeft = NaN
  let lastRight = NaN

  const ro = new ResizeObserver(() => measure())

  const measure = () => {
    const next = document.querySelector<HTMLElement>('[data-session-anchor="workspace"]')

    if (next !== el) {
      if (el) {
        ro.unobserve(el)
      }

      el = next

      if (el) {
        ro.observe(el)
      }
    }

    if (!el) {
      return
    }

    const r = el.getBoundingClientRect()
    const left = Math.round(r.left)
    const right = Math.round(window.innerWidth - r.right)

    // Skip unchanged writes: a sash drag fires the RO every frame, and each
    // :root custom-property set dirties style for everything that reads them.
    if (left !== lastLeft) {
      lastLeft = left
      root.style.setProperty('--workspace-left', `${left}px`)
    }

    if (right !== lastRight) {
      lastRight = right
      root.style.setProperty('--workspace-right', `${right}px`)
    }
  }

  // Tree mutations move zones without resizing them (⌘\ flip) — re-measure a
  // frame later, after the DOM committed. RO covers width changes (sash drags,
  // side collapses); window resize covers the rest.
  const unsubTree = $layoutTree.listen(() => requestAnimationFrame(measure))
  window.addEventListener('resize', measure)
  measure()

  return () => {
    unsubTree()
    window.removeEventListener('resize', measure)
    ro.disconnect()
    root.style.removeProperty('--workspace-left')
    root.style.removeProperty('--workspace-right')
  }
}

/**
 * Intersects an element's live viewport rect with the native window-controls
 * rect. Returns the overlap in ELEMENT-LOCAL coordinates (null when clear), so
 * the consumer can reserve the space and paint a drag strip without knowing
 * anything about platform, fullscreen state, or layout position.
 */
export function useWindowControlsOverlap(ref: RefObject<HTMLElement | null>, enabled = true): Rect | null {
  const controls = useWindowControlsRect()
  const [overlap, setOverlap] = useState<Rect | null>(null)

  useLayoutEffect(() => {
    const el = ref.current

    if (!enabled || !controls || !el) {
      setOverlap(null)

      return
    }

    const update = () => {
      const r = el.getBoundingClientRect()
      const hit = intersect(controls, { x: r.x, y: r.y, width: r.width, height: r.height })
      const local = hit ? { x: hit.x - r.x, y: hit.y - r.y, width: hit.width, height: hit.height } : null

      setOverlap(prev => (sameRect(prev, local) ? prev : local))
    }

    update()

    // Size changes fire the observer; cross-window moves fire `resize`. A pane
    // shifted only by a sibling's resize re-measures on its own grid reflow
    // (its track width changes), so this covers the shell's real cases.
    const ro = new ResizeObserver(update)
    ro.observe(el)
    window.addEventListener('resize', update)

    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [controls, enabled, ref])

  return overlap
}
