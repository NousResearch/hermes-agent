import { useEffect, useState } from 'react'

// Measure the EXACT footprint of the native window-controls overlay
// (min/max/close) straight from the browser, instead of a hardcoded
// reservation.
//
// When Electron's Window Controls Overlay is active (native Windows AND Linux),
// Chromium exposes `navigator.windowControlsOverlay`. Its getTitlebarAreaRect()
// returns the draggable title-bar rect that EXCLUDES the controls, so the
// controls' footprint is everything outside it. Chromium follows the host's
// button layout, so the same app meets the controls on the RIGHT (Windows
// default) or on the LEFT (macOS-style KDE/GTK layouts, e.g. KWin
// `ButtonsOnLeft`). Measuring both edges keeps the clusters clear of them
// across DPI / host themes / window states — no magic numbers.
//
// Returns null when WCO is unavailable (macOS, or before first layout), so
// callers fall back to the static reservation from the main process.

interface WindowControlsOverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

export interface WindowControlsInsets {
  left: number
  right: number
}

const overlay = () =>
  (navigator as Navigator & { windowControlsOverlay?: WindowControlsOverlayLike }).windowControlsOverlay ?? null

function measure(wco: WindowControlsOverlayLike | null, viewportWidth: number): WindowControlsInsets | null {
  const rect = wco?.visible ? wco.getTitlebarAreaRect() : null

  // No overlay, or it isn't laid out yet.
  if (!rect?.width) {
    return null
  }

  return {
    left: Math.max(0, Math.round(rect.left)),
    right: Math.max(0, Math.round(viewportWidth - rect.right))
  }
}

/** Exported for tests: the pure measurement behind the hook. */
export function measureWindowControlsInsets(
  wco: WindowControlsOverlayLike | null,
  viewportWidth: number
): WindowControlsInsets | null {
  return measure(wco, viewportWidth)
}

/**
 * Live insets (px) of the native window-controls overlay on each titlebar edge,
 * or null when the platform/build exposes no overlay (caller should use the
 * static fallback).
 */
export function useWindowControlsOverlayInsets(): WindowControlsInsets | null {
  const [insets, setInsets] = useState<WindowControlsInsets | null>(() =>
    measure(overlay(), typeof window === 'undefined' ? 0 : window.innerWidth)
  )

  useEffect(() => {
    const wco = overlay()

    if (!wco) {
      return
    }

    const update = () => setInsets(measure(wco, window.innerWidth))

    // Re-measure on overlay geometry changes (maximize/restore, DPI) and on
    // window resize (innerWidth feeds the calc).
    wco.addEventListener('geometrychange', update)
    window.addEventListener('resize', update)
    update()

    return () => {
      wco.removeEventListener('geometrychange', update)
      window.removeEventListener('resize', update)
    }
  }, [])

  return insets
}
