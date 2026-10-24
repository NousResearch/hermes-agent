import { useEffect, useState } from 'react'

// Measure the EXACT width of the native window-controls overlay (min/max/close)
// straight from the browser, instead of a hardcoded reservation.
//
// When Electron's Window Controls Overlay is active (native Windows AND WSLg),
// Chromium exposes `navigator.windowControlsOverlay`. Its getTitlebarAreaRect()
// returns the draggable title-bar rect that EXCLUDES the controls, so the
// controls' width on the right is `innerWidth - rect.right` and on the left is
// `rect.left` (Linux honours the desktop's button order, so the controls may sit
// on either side or both). This is precise and self-correcting across DPI / host
// themes / window states — no magic numbers, and it sidesteps the WSLg-vs-Windows
// footprint guesswork.
//
// Returns null when WCO is unavailable (macOS or before first layout), so callers
// fall back to the static reservation from the main process.

export interface WindowControlsOverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

const overlay = () =>
  (navigator as Navigator & { windowControlsOverlay?: WindowControlsOverlayLike }).windowControlsOverlay ?? null

export interface OverlayInsets {
  left: number
  right: number
}

export function measureOverlayInsets(wco: WindowControlsOverlayLike | null, innerWidth: number): OverlayInsets | null {
  const rect = wco?.visible ? wco.getTitlebarAreaRect() : null

  // No overlay, or it isn't laid out yet.
  if (!rect?.width) {
    return null
  }

  return {
    left: Math.max(0, Math.round(rect.left)),
    right: Math.max(0, Math.round(innerWidth - rect.right))
  }
}

const sameInsets = (a: OverlayInsets | null, b: OverlayInsets | null) => a?.left === b?.left && a?.right === b?.right

/**
 * Live widths (px) the native window-controls overlay takes on each side, or
 * null when the platform/build exposes no overlay (caller should use the static
 * fallback).
 */
export function useWindowControlsOverlayInsets(): OverlayInsets | null {
  const [insets, setInsets] = useState<OverlayInsets | null>(() => measureOverlayInsets(overlay(), window.innerWidth))

  useEffect(() => {
    const wco = overlay()

    if (!wco) {
      return
    }

    const update = () => setInsets(prev => {
      const next = measureOverlayInsets(wco, window.innerWidth)

      return sameInsets(prev, next) ? prev : next
    })

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
