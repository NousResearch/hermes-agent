/**
 * Pane geometry — AABB INTERSECTION → WINDOW-CONTROL AWARENESS. The native
 * window controls (macOS traffic lights top-left / Windows overlay or Linux
 * native overlay on either side) are rectangles in viewport pixels. Any region
 * whose rect intersects one must reserve that space and expose a drag strip.
 * One `intersect()` call replaces per-layout inset special cases.
 *
 * Also publishes the WORKSPACE zone's edges as CSS vars (see
 * publishWorkspaceGeometry) — chrome that aligns to the main pane (titlebar
 * title et al) consumes `var(--workspace-left/right)` in plain CSS instead of
 * threading rects through React.
 */

import { type RefObject, useLayoutEffect, useState, useSyncExternalStore } from 'react'

import { $layoutTree } from '@/components/pane-shell/tree/store'
import { $connection } from '@/store/session'

import { OVERLAY_BUTTON_WIDTH } from '../../../electron/titlebar-overlay-width'

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
const MACOS_FALLBACK_BUTTON_X = 12

interface ConnectionLike {
  windowButtonPosition?: { x: number; y: number } | null
  nativeOverlayWidth?: number | null
  linuxWindowButtons?: { left: number; right: number } | null
  isFullscreen?: boolean | null
}

/**
 * Native window-control rectangles in viewport pixels. Empty when there is
 * nothing to dodge (fullscreen, plain browser, secondary windows with hidden
 * controls).
 */
export function windowControlsRects(connection: ConnectionLike | null, viewportWidth: number): Rect[] {
  const inElectron = typeof window !== 'undefined' && 'hermesDesktop' in window

  if (!inElectron) {
    return []
  }

  if (connection?.isFullscreen) {
    return []
  }

  const rects: Rect[] = []
  const buttons = connection?.linuxWindowButtons

  if (buttons) {
    // Plain Linux: GNOME/GTK's button-layout decides which side the native
    // overlay buttons occupy. Counts come from main's gsettings read.
    const left = Math.max(0, buttons.left) * OVERLAY_BUTTON_WIDTH
    const right = Math.max(0, buttons.right) * OVERLAY_BUTTON_WIDTH

    if (left > 0) {
      rects.push({ x: 0, y: 0, width: left, height: CONTROLS_BAND_HEIGHT })
    }

    if (right > 0) {
      rects.push({ x: viewportWidth - right, y: 0, width: right, height: CONTROLS_BAND_HEIGHT })
    }
  } else {
    // Windows / WSLg: the whole native overlay sits on the top-right.
    const overlayWidth = connection?.nativeOverlayWidth ?? 0

    if (overlayWidth > 0) {
      rects.push({ x: viewportWidth - overlayWidth, y: 0, width: overlayWidth, height: CONTROLS_BAND_HEIGHT })
    }
  }

  if (rects.length > 0) {
    return rects
  }

  // macOS: traffic lights on the top-left. windowButtonPosition === null means
  // the platform has no left-side controls at all.
  const pos = connection?.windowButtonPosition

  if (pos === null) {
    return []
  }

  const isMac = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform)

  if (!pos && !isMac) {
    return []
  }

  const x = pos?.x ?? MACOS_FALLBACK_BUTTON_X

  return [{ x: 0, y: 0, width: x + MACOS_LIGHTS_WIDTH, height: CONTROLS_BAND_HEIGHT }]
}

/** Bounding box of two overlapping rectangles. */
function unionRect(a: Rect, b: Rect): Rect {
  const x = Math.min(a.x, b.x)
  const y = Math.min(a.y, b.y)
  const right = Math.max(a.x + a.width, b.x + b.width)
  const bottom = Math.max(a.y + a.height, b.y + b.height)

  return { x, y, width: right - x, height: bottom - y }
}

// ---------------------------------------------------------------------------
// Live hook
// ---------------------------------------------------------------------------

const NO_CONTROLS: Rect[] = []
let cachedRects: Rect[] = NO_CONTROLS
let cachedKey = ''

function rectsKey(rects: Rect[]) {
  return rects.map(r => `${r.x},${r.y},${r.width},${r.height}`).join('|')
}

function readControlsRects(): Rect[] {
  const next = windowControlsRects($connection.get(), typeof window === 'undefined' ? 0 : window.innerWidth)
  const key = rectsKey(next)

  // Referentially stable snapshot for useSyncExternalStore.
  if (key !== cachedKey) {
    cachedKey = key
    cachedRects = next
  }

  return cachedRects
}

function subscribeControlsRects(cb: () => void) {
  const unsubConnection = $connection.subscribe(() => cb())
  window.addEventListener('resize', cb)

  return () => {
    unsubConnection()
    window.removeEventListener('resize', cb)
  }
}

/** Reactive native window-control rects (connection + viewport aware). */
export function useWindowControlsRects(): Rect[] {
  return useSyncExternalStore(subscribeControlsRects, readControlsRects, () => NO_CONTROLS)
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

// --- Sash-drag deferral ------------------------------------------------------
// The tree sash sets this for the duration of a resize gesture (pointerdown to
// pointerup). While set, `publishWorkspaceGeometry` skips its :root custom
// property writes: each one invalidates computed style for the whole document,
// and the sash's ResizeObserver fires per frame. Measured live (LoAF, real
// session): drag frames of ~68ms with style+layout=67ms and no script ≥5ms;
// suppressing the writes recovered 14fps → 51fps. The vars only align titlebar
// chrome — republishing once on release is visually identical.
let sashDragDepth = 0
let onSashDragEnd: null | (() => void) = null

export function beginSashDrag() {
  sashDragDepth += 1
}

export function endSashDrag() {
  sashDragDepth = Math.max(0, sashDragDepth - 1)

  if (sashDragDepth === 0) {
    onSashDragEnd?.()
  }
}

const sashDragging = () => sashDragDepth > 0

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
    // DEFER during a sash drag (see beginSashDrag above) — republished once on
    // release via the onSashDragEnd hook registered below.
    if (sashDragging()) {
      return
    }

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
  // Drag released → publish the final geometry the deferral above skipped.
  onSashDragEnd = () => requestAnimationFrame(measure)
  window.addEventListener('resize', measure)
  measure()

  return () => {
    unsubTree()
    onSashDragEnd = null
    window.removeEventListener('resize', measure)
    ro.disconnect()
    root.style.removeProperty('--workspace-left')
    root.style.removeProperty('--workspace-right')
  }
}

/**
 * Intersects an element's live viewport rect with the native window-control
 * rectangles. Returns their combined overlap in ELEMENT-LOCAL coordinates
 * (null when clear), so the consumer can reserve the space and paint a drag
 * strip without knowing anything about platform, fullscreen state, side, or
 * layout position.
 */
export function useWindowControlsOverlap(ref: RefObject<HTMLElement | null>, enabled = true): Rect | null {
  const controls = useWindowControlsRects()
  const [overlap, setOverlap] = useState<Rect | null>(null)

  useLayoutEffect(() => {
    const el = ref.current

    if (!enabled || controls.length === 0 || !el) {
      setOverlap(null)

      return
    }

    const update = () => {
      const r = el.getBoundingClientRect()
      const viewport = { x: r.x, y: r.y, width: r.width, height: r.height }
      let local: Rect | null = null

      for (const control of controls) {
        const hit = intersect(control, viewport)

        if (!hit) {
          continue
        }

        const next = { x: hit.x - r.x, y: hit.y - r.y, width: hit.width, height: hit.height }

        local = local ? unionRect(local, next) : next
      }

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
