import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useCallback,
  useLayoutEffect,
  useRef,
  useState
} from 'react'

import { isSmartZoomWheel } from '@/lib/trackpad-gestures'

interface Transform {
  scale: number
  x: number
  y: number
}

const MIN_SCALE = 0.05
const MAX_SCALE = 8
const WHEEL_STEP = 1.1
const BUTTON_STEP = 1.25

const clamp = (scale: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))

/**
 * Headless pan/zoom transform. Wheel zooms toward the cursor, drag pans, buttons
 * zoom toward centre, and the initial view fits the content into the stage
 * (never upscaling). Returns the transform style plus the surface handlers, so
 * any content (SVG, image, canvas) can be made pan/zoomable.
 */
export function useZoomPan() {
  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number } | null>(null)
  const [panning, setPanning] = useState(false)
  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null)
  const [contentEl, setContentEl] = useState<HTMLDivElement | null>(null)

  // Zoom toward (cx, cy), measured from the surface centre, keeping that point fixed.
  const zoomAt = useCallback((factor: number, cx = 0, cy = 0) => {
    setTransform(prev => {
      const scale = clamp(prev.scale * factor)
      const k = scale / prev.scale

      return { scale, x: cx - k * (cx - prev.x), y: cy - k * (cy - prev.y) }
    })
  }, [])

  // Shrink the content so it fits the stage, never upscale it. The stage grid
  // centers the content, so the fit transform needs no translation. Content
  // with no measurable size yet (async render, e.g. mermaid) stays as-is — a
  // zero scale would blank the overlay instead of waiting for geometry.
  const fit = useCallback(() => {
    if (!stageEl || !contentEl) {
      return
    }

    const availableW = stageEl.clientWidth
    const availableH = stageEl.clientHeight
    const contentW = contentEl.scrollWidth
    const contentH = contentEl.scrollHeight

    if (availableW <= 0 || availableH <= 0 || contentW <= 0 || contentH <= 0) {
      return
    }

    const scale = Math.min(availableW / contentW, availableH / contentH, 1)

    setTransform({ scale: clamp(scale), x: 0, y: 0 })
  }, [contentEl, stageEl])

  // Manual zoom/pan opts out of refitting until reset; while the view is still
  // the fitted one, stage or content resizes (dialog resize, async SVG
  // appearing, window resize) re-fit instead of stranding a zoomed view.
  const fittedRef = useRef(true)

  const fitIfFitted = useCallback(() => {
    if (fittedRef.current) {
      fit()
    }
  }, [fit])

  // The overlay lives in a portal that mounts its DOM in a later commit than
  // the hook consumer, so node state (not object refs) drives the
  // subscription: attaching the nodes re-runs this effect and the fit rides
  // the observer's spec-guaranteed first delivery once geometry exists.
  useLayoutEffect(() => {
    if (!stageEl || !contentEl || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(fitIfFitted)

    observer.observe(stageEl)
    observer.observe(contentEl)

    return () => observer.disconnect()
  }, [contentEl, fitIfFitted, stageEl])

  const zoomIn = useCallback(() => {
    fittedRef.current = false
    zoomAt(BUTTON_STEP)
  }, [zoomAt])

  const zoomOut = useCallback(() => {
    fittedRef.current = false
    zoomAt(1 / BUTTON_STEP)
  }, [zoomAt])

  const reset = useCallback(() => {
    fittedRef.current = true
    fit()
  }, [fit])

  const onWheel = useCallback(
    (event: ReactWheelEvent) => {
      event.preventDefault()

      // macOS smart zoom (two-finger double-tap) → fitted view, not zoom-in.
      if (isSmartZoomWheel(event)) {
        reset()

        return
      }

      fittedRef.current = false

      const rect = event.currentTarget.getBoundingClientRect()
      const cx = event.clientX - rect.left - rect.width / 2
      const cy = event.clientY - rect.top - rect.height / 2

      zoomAt(event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, cx, cy)
    },
    [reset, zoomAt]
  )

  const onPointerDown = useCallback((event: ReactPointerEvent) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    fittedRef.current = false
    setTransform(prev => {
      drag.current = { x: event.clientX - prev.x, y: event.clientY - prev.y }

      return prev
    })
    setPanning(true)
  }, [])

  const onPointerMove = useCallback((event: ReactPointerEvent) => {
    if (!drag.current) {
      return
    }

    const start = drag.current

    setTransform(prev => ({ ...prev, x: event.clientX - start.x, y: event.clientY - start.y }))
  }, [])

  const endPan = useCallback(() => {
    drag.current = null
    setPanning(false)
  }, [])

  const style: CSSProperties = {
    transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`
  }

  return {
    panning,
    reset,
    scale: transform.scale,
    setContentEl,
    setStageEl,
    stageProps: { onPointerDown, onPointerLeave: endPan, onPointerMove, onPointerUp: endPan, onWheel },
    style,
    zoomIn,
    zoomOut
  }
}
