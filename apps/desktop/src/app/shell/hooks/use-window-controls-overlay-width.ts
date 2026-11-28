import { useEffect, useState } from 'react'

import { OVERLAY_BUTTON_WIDTH } from '../../../../electron/titlebar-overlay-width'

// Measure the EXACT native window-controls reservation straight from the
// browser, instead of a hardcoded footprint.
//
// When Electron's Window Controls Overlay is active, Chromium exposes
// `navigator.windowControlsOverlay`. Its getTitlebarAreaRect() returns the
// draggable title-bar rect that EXCLUDES the controls, so the controls on the
// left are `rect.x` wide and the ones on the right are
// `innerWidth - (rect.x + rect.width)`. This is precise and self-correcting
// across DPI / host themes / window states.
//
// Returns null when WCO is unavailable (macOS, WSLg, before first layout), so
// callers fall back to GNOME's button-layout counts from the main process.

export interface WindowControlsReservation {
  left: number
  right: number
}

interface WindowControlsOverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

const overlay = () =>
  (navigator as Navigator & { windowControlsOverlay?: WindowControlsOverlayLike }).windowControlsOverlay ?? null

function measure(wco: WindowControlsOverlayLike | null): WindowControlsReservation | null {
  const rect = wco?.visible ? wco.getTitlebarAreaRect() : null

  // No overlay, or it isn't laid out yet.
  if (!rect?.width) {
    return null
  }

  const left = Math.max(0, Math.round(rect.x))
  const right = Math.max(0, Math.round(window.innerWidth - (rect.x + rect.width)))

  return left > 0 || right > 0 ? { left, right } : null
}

/**
 * Static reservation from GNOME's button-layout counts. Plain Linux always has
 * `linuxWindowButtons`; Windows/WSLg use the whole right-side overlay fallback.
 */
export function windowControlsFallback(
  linuxWindowButtons: { left: number; right: number } | null | undefined,
  nativeOverlayWidth: number
): WindowControlsReservation {
  if (linuxWindowButtons) {
    return {
      left: Math.max(0, linuxWindowButtons.left) * OVERLAY_BUTTON_WIDTH,
      right: Math.max(0, linuxWindowButtons.right) * OVERLAY_BUTTON_WIDTH
    }
  }

  return { left: 0, right: Math.max(0, nativeOverlayWidth) }
}

/**
 * Live native window-control reservation (px on each side), or null when the
 * platform/build exposes no WCO (caller should use `windowControlsFallback`).
 */
export function useWindowControlsOverlayWidth(): WindowControlsReservation | null {
  const [reservation, setReservation] = useState<WindowControlsReservation | null>(() => measure(overlay()))

  useEffect(() => {
    const wco = overlay()

    if (!wco) {
      return
    }

    const update = () => setReservation(measure(wco))

    // Re-measure on overlay geometry changes (maximize/restore, DPI/button
    // layout) and on window resize (innerWidth feeds the calc).
    wco.addEventListener('geometrychange', update)
    window.addEventListener('resize', update)
    update()

    return () => {
      wco.removeEventListener('geometrychange', update)
      window.removeEventListener('resize', update)
    }
  }, [])

  return reservation
}
