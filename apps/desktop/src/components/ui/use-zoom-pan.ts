import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
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

const MIN_SCALE = 0.25
const MAX_SCALE = 8
const WHEEL_STEP = 1.1
const BUTTON_STEP = 1.25
// Breathing room around fitted content. Vertical clears the floating toolbar
// at the stage's bottom edge; symmetric so the grid centering stays exact.
const FIT_INSET_X = 32
const FIT_INSET_Y = 64
// A fitted diagram can shrink far below the interactive zoom-out floor, so
// surfaces that fit content (see `setContentEl`) pass this via `minScale`.
export const FIT_MIN_SCALE = 0.05
// Pointer travel (px) below which a single-pointer gesture counts as a click
// rather than a pan. Lets consumers (e.g. an image lightbox) close on a clean
// click while still panning after a real drag.
const DRAG_THRESHOLD = 4

interface UseZoomPanOptions {
  enabled?: boolean
  maxScale?: number
  minScale?: number
}

/**
 * Headless pan/zoom transform shared by every zoomable surface (image lightbox,
 * diagram/artifact viewer, …). Wheel zooms toward the cursor, drag pans,
 * two-finger pinch zooms + pans, and the +/- buttons zoom toward centre. When
 * a content element is registered via `setContentEl`, the initial view fits
 * the content into the stage (never upscaling).
 *
 * The wheel listener is attached natively (non-passive) so `preventDefault`
 * actually stops the page/dialog from scrolling underneath, which a React
 * `onWheel` (passive at the root) cannot do.
 *
 * `moved` reports whether the current gesture moved (pan/pinch) vs. a clean
 * click, so callers can gate click-to-dismiss on it.
 */
export function useZoomPan<T extends HTMLElement = HTMLElement>(options: UseZoomPanOptions = {}) {
  const { enabled = true, minScale = MIN_SCALE, maxScale = MAX_SCALE } = options
  const clamp = useCallback((scale: number) => Math.min(maxScale, Math.max(minScale, scale)), [minScale, maxScale])

  const ref = useRef<T>(null)
  // The surface node as state, not just a ref: the surface may mount AFTER
  // `enabled` flips (the lightbox img renders inside a dialog portal, a commit
  // later than the open flag). An effect keyed on [enabled] alone would capture
  // a null ref and never attach the native wheel listener. Re-rendering on
  // mount re-runs that effect with the node in hand. It doubles as the fit
  // stage when a content element is registered.
  const [node, setNode] = useState<T | null>(null)

  const refCallback = useCallback((instance: T | null) => {
    ref.current = instance
    setNode(instance)
  }, [])

  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [moved, setMoved] = useState(false)

  // Track active pointers for drag (1) and pinch (2), plus the in-flight drag
  // anchor and the pinch baseline.
  const drag = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const pinch = useRef<{ dist: number; midX: number; midY: number } | null>(null)

  // The content wrapper the fit path measures (see fit). Null on surfaces
  // that zoom their only child directly (image lightbox), which opt out of
  // fitting.
  const [contentEl, setContentEl] = useState<HTMLDivElement | null>(null)

  // Zoom toward (cx, cy), measured from the surface centre, keeping that point fixed.
  const zoomAt = useCallback(
    (factor: number, cx = 0, cy = 0) => {
      setTransform(prev => {
        const scale = clamp(prev.scale * factor)
        const k = scale / prev.scale

        return { scale, x: cx - k * (cx - prev.x), y: cy - k * (cy - prev.y) }
      })
    },
    [clamp]
  )

  // Shrink the content so it fits the stage, inset from its edges, and never
  // upscale it. The stage grid centers the content, so the fit transform needs
  // no translation. Content with no measurable size yet (async render, e.g.
  // mermaid) stays as-is — a zero scale would blank the overlay instead of
  // waiting for geometry.
  const fit = useCallback(() => {
    if (!node || !contentEl) {
      return
    }

    const availableW = node.clientWidth - FIT_INSET_X * 2
    const availableH = node.clientHeight - FIT_INSET_Y * 2
    const contentW = contentEl.scrollWidth
    const contentH = contentEl.scrollHeight

    if (availableW <= 0 || availableH <= 0 || contentW <= 0 || contentH <= 0) {
      return
    }

    const scale = Math.min(availableW / contentW, availableH / contentH, 1)

    setTransform({ scale: clamp(scale), x: 0, y: 0 })
  }, [clamp, contentEl, node])

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
    if (!node || !contentEl || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(fitIfFitted)

    observer.observe(node)
    observer.observe(contentEl)

    return () => observer.disconnect()
  }, [contentEl, fitIfFitted, node])

  const reset = useCallback(() => {
    setTransform({ scale: 1, x: 0, y: 0 })
    setMoved(false)
    setPanning(false)
    fittedRef.current = true
    // Refit when a content element is registered; surfaces without one (the
    // image lightbox) keep the identity view.
    fit()
  }, [fit])

  const zoomIn = useCallback(() => {
    fittedRef.current = false

    const node = ref.current
    const rect = node?.getBoundingClientRect()
    zoomAt(BUTTON_STEP, rect ? rect.width / 2 : 0, rect ? rect.height / 2 : 0)
  }, [zoomAt])

  const zoomOut = useCallback(() => {
    fittedRef.current = false

    const node = ref.current
    const rect = node?.getBoundingClientRect()
    zoomAt(1 / BUTTON_STEP, rect ? rect.width / 2 : 0, rect ? rect.height / 2 : 0)
  }, [zoomAt])

  // Native, non-passive wheel so we can preventDefault page scroll. Attached to
  // the surface node (ref) only while the viewer is enabled, so it never
  // hijacks wheel events when the lightbox/dialog is closed. The handler's
  // `fittedRef.current = false` is not an atom-mirror — a one-way gesture flag
  // marking the view as manually zoomed.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (!node || !enabled) {
      return
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()

      // macOS smart zoom (two-finger double-tap) → fitted view, not zoom-in.
      if (isSmartZoomWheel(event)) {
        reset()

        return
      }

      fittedRef.current = false

      const rect = node.getBoundingClientRect()
      const cx = event.clientX - rect.left - rect.width / 2
      const cy = event.clientY - rect.top - rect.height / 2

      zoomAt(event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, cx, cy)
    }

    node.addEventListener('wheel', onWheel, { passive: false })

    return () => node.removeEventListener('wheel', onWheel)
  }, [enabled, node, reset, zoomAt])

  const endPan = useCallback(() => {
    drag.current = null
    setPanning(false)
  }, [])

  const onPointerDown = useCallback((event: ReactPointerEvent<T>) => {
    event.currentTarget.setPointerCapture?.(event.pointerId)
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    setMoved(false)
    fittedRef.current = false

    if (pointers.current.size === 1) {
      drag.current = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY }
      pinch.current = null
    } else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()]
      pinch.current = {
        dist: Math.hypot(a.x - b.x, a.y - b.y),
        midX: (a.x + b.x) / 2,
        midY: (a.y + b.y) / 2
      }
      drag.current = null
    }
  }, [])

  const onPointerMove = useCallback(
    (event: ReactPointerEvent<T>) => {
      if (!pointers.current.has(event.pointerId)) {
        return
      }

      pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY })

      const node = ref.current

      if (!node) {
        return
      }

      const rect = node.getBoundingClientRect()
      const originX = rect.width / 2
      const originY = rect.height / 2

      if (pointers.current.size >= 2 && pinch.current) {
        const [a, b] = [...pointers.current.values()]
        const dist = Math.hypot(a.x - b.x, a.y - b.y)
        const midX = (a.x + b.x) / 2
        const midY = (a.y + b.y) / 2
        const factor = dist / (pinch.current.dist || dist)

        setTransform(prev => {
          const scale = clamp(prev.scale * factor)
          const k = scale / prev.scale
          // Focal point relative to the surface centre; zoom about it, then add
          // the midpoint translation so the gesture follows the fingers.
          const fx = midX - rect.left - originX
          const fy = midY - rect.top - originY

          return {
            scale,
            x: prev.x + (midX - pinch.current!.midX) + (fx - k * fx),
            y: prev.y + (midY - pinch.current!.midY) + (fy - k * fy)
          }
        })

        pinch.current = { dist, midX, midY }
        setMoved(true)

        return
      }

      if (pointers.current.size === 1 && drag.current) {
        const dx = event.clientX - drag.current.x
        const dy = event.clientY - drag.current.y

        if (Math.hypot(event.clientX - drag.current.startX, event.clientY - drag.current.startY) > DRAG_THRESHOLD) {
          setMoved(true)
        }

        setTransform(prev => ({ ...prev, x: prev.x + dx, y: prev.y + dy }))
        drag.current = { ...drag.current, x: event.clientX, y: event.clientY }
      }
    },
    [clamp]
  )

  const onPointerUp = useCallback((event: ReactPointerEvent<T>) => {
    event.currentTarget.releasePointerCapture?.(event.pointerId)
    pointers.current.delete(event.pointerId)
    setPanning(false)

    if (pointers.current.size === 1) {
      // Lifting one finger of a pinch continues as a single-finger pan.
      const [only] = [...pointers.current.values()]
      drag.current = { x: only.x, y: only.y, startX: only.x, startY: only.y }
      setMoved(true)
      pinch.current = null
    } else if (pointers.current.size === 0) {
      drag.current = null
      pinch.current = null
    }
  }, [])

  // A canceled pointer (the browser steals the gesture for scroll/selection,
  // or a touch is interrupted) must get the same cleanup as completion. Without
  // it the stale entry in `pointers` makes the next pointerdown look like a
  // pinch with old coordinates.
  const onPointerCancel = useCallback((event: ReactPointerEvent<T>) => {
    event.currentTarget.releasePointerCapture?.(event.pointerId)
    pointers.current.delete(event.pointerId)

    if (pointers.current.size <= 1) {
      drag.current = null
      pinch.current = null
      setPanning(false)
      setMoved(false)
    }
  }, [])

  const style: CSSProperties = {
    transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`
  }

  return {
    moved,
    panning,
    ref: refCallback,
    reset,
    scale: transform.scale,
    setContentEl,
    stageProps: {
      onPointerCancel,
      onPointerDown,
      onPointerLeave: endPan,
      onPointerMove,
      onPointerUp
    },
    style,
    zoomIn,
    zoomOut
  }
}
