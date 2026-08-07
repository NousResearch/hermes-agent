import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useCallback,
  useEffect,
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
 * two-finger pinch zooms + pans, and the +/- buttons zoom toward centre.
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
  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [moved, setMoved] = useState(false)

  // Track active pointers for drag (1) and pinch (2), plus the in-flight drag
  // anchor and the pinch baseline.
  const drag = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const pinch = useRef<{ dist: number; midX: number; midY: number } | null>(null)

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

  const reset = useCallback(() => {
    setTransform({ scale: 1, x: 0, y: 0 })
    setMoved(false)
    setPanning(false)
  }, [])

  const zoomIn = useCallback(() => {
    const node = ref.current
    const rect = node?.getBoundingClientRect()
    zoomAt(BUTTON_STEP, rect ? rect.width / 2 : 0, rect ? rect.height / 2 : 0)
  }, [zoomAt])

  const zoomOut = useCallback(() => {
    const node = ref.current
    const rect = node?.getBoundingClientRect()
    zoomAt(1 / BUTTON_STEP, rect ? rect.width / 2 : 0, rect ? rect.height / 2 : 0)
  }, [zoomAt])

  // Native, non-passive wheel so we can preventDefault page scroll. Attached to
  // the surface node (ref) only while the viewer is enabled, so it never
  // hijacks wheel events when the lightbox/dialog is closed.
  useEffect(() => {
    const node = ref.current
    if (!node || !enabled) {
      return
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()

      // macOS smart zoom (two-finger double-tap) → reset, not zoom-in.
      if (isSmartZoomWheel(event)) {
        setTransform({ scale: 1, x: 0, y: 0 })
        setMoved(false)

        return
      }

      const rect = node.getBoundingClientRect()
      const cx = event.clientX - rect.left - rect.width / 2
      const cy = event.clientY - rect.top - rect.height / 2

      zoomAt(event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, cx, cy)
    }

    node.addEventListener('wheel', onWheel, { passive: false })

    return () => node.removeEventListener('wheel', onWheel)
  }, [enabled, zoomAt])

  const endPan = useCallback(() => {
    drag.current = null
    setPanning(false)
  }, [])

  const onPointerDown = useCallback((event: ReactPointerEvent<T>) => {
    event.currentTarget.setPointerCapture?.(event.pointerId)
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    setMoved(false)

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
    ref,
    reset,
    scale: transform.scale,
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
