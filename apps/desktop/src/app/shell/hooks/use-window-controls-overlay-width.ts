import { useEffect, useState } from 'react'

// Measure the EXACT width of the native window-controls overlay (min/max/close)
// straight from the browser, instead of a hardcoded reservation.
//
// When Electron's Window Controls Overlay is active (native Windows AND WSLg),
// Chromium exposes `navigator.windowControlsOverlay`. Its getTitlebarAreaRect()
// returns the draggable title-bar rect that EXCLUDES the controls, so the
// controls' width on the right is `innerWidth - rect.right`. This is precise and
// self-correcting across DPI / host themes / window states — no magic numbers,
// and it sidesteps the WSLg-vs-Windows footprint guesswork.
//
// Returns null when WCO is unavailable (macOS, plain Linux, or before first
// layout), so callers fall back to the static reservation from the main process.

interface WindowControlsOverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

const overlay = () =>
  (navigator as Navigator & { windowControlsOverlay?: WindowControlsOverlayLike }).windowControlsOverlay ?? null

function measure(wco: WindowControlsOverlayLike | null): number | null {
  const rect = wco?.visible ? wco.getTitlebarAreaRect() : null

  // No overlay, or it isn't laid out yet.
  if (!rect?.width) {
    return null
  }

  const width = Math.round(window.innerWidth - rect.right)

  return width > 0 ? width : null
}

/**
 * Live width (px) of the right-side native window-controls overlay, or null when
 * the platform/build exposes no overlay (caller should use the static fallback).
 */
export function useWindowControlsOverlayWidth(): number | null {
  const [width, setWidth] = useState<number | null>(() => measure(overlay()))

  useEffect(() => {
    const wco = overlay()

    if (!wco) {
      return
    }

    const update = () => setWidth(measure(wco))

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

  return width
}
