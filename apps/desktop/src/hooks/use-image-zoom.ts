'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  MAX_ZOOM,
  MIN_ZOOM,
  type Point,
  ZOOM_STEP,
  clampPan,
  clampZoom,
  isTypingTarget,
  zoomAtPoint,
  zoomFromWheel
} from '@/lib/image-zoom'

const ORIGIN: Point = { x: 0, y: 0 }

/**
 * Zoom + pan state for the lightbox: wheel, drag, keyboard, double-click.
 *
 * Pointer work stays in refs and is written to the DOM directly during a drag —
 * a React state update per pointermove would re-render the dialog on every
 * frame of a gesture that is pure transform.
 *
 * `vector` picks HOW the zoom is applied. A raster is magnified with
 * `transform: scale()`, which stretches the bitmap the browser already
 * rasterized. Doing that to an SVG blurs a diagram that has the data to stay
 * sharp, so a vector grows its LAYOUT box (width/height) and the engine
 * re-renders it at the new size; pan stays a translate either way.
 */
export function useImageZoom(active: boolean, vector = false, modal = false) {
  const [scale, setScale] = useState(MIN_ZOOM)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const hoverRef = useRef(false)
  const offsetRef = useRef<Point>(ORIGIN)
  const dragRef = useRef<{ origin: Point; pointerId: number; start: Point } | null>(null)
  // The laid-out size at fit, captured before any zoom scales the box. For a
  // vector this is the baseline every zoom level multiplies.
  const naturalRef = useRef<{ height: number; width: number } | null>(null)

  const captureNatural = useCallback(() => {
    const node = imageRef.current
    if (!node) return null
    if (!naturalRef.current && node.offsetWidth > 0) {
      naturalRef.current = { height: node.offsetHeight, width: node.offsetWidth }
    }
    return naturalRef.current
  }, [])

  const paint = useCallback(
    (next: Point, nextScale: number) => {
      const node = imageRef.current
      if (!node) return
      if (vector) {
        const natural = captureNatural()
        if (natural) {
          // Re-layout at the new size: the renderer redraws the vector sharp.
          node.style.width = `${natural.width * nextScale}px`
          node.style.height = `${natural.height * nextScale}px`
          node.style.maxWidth = nextScale > MIN_ZOOM ? 'none' : ''
          node.style.maxHeight = nextScale > MIN_ZOOM ? 'none' : ''
        }
        node.style.transform = `translate3d(${next.x}px, ${next.y}px, 0)`
        return
      }
      node.style.transform = `translate3d(${next.x}px, ${next.y}px, 0) scale(${nextScale})`
    },
    [captureNatural, vector]
  )

  const baseSize = useCallback((): { height: number; width: number } => {
    const node = imageRef.current
    if (!node) return { height: 0, width: 0 }
    // The pan bounds are computed from the FIT size in both modes: for a vector
    // the node has already grown, so reading it back would square the scale.
    if (vector) return captureNatural() ?? { height: node.offsetHeight, width: node.offsetWidth }
    // offsetWidth/Height are the *laid-out* box, unaffected by the transform —
    // reading getBoundingClientRect here would compound the current scale.
    return { height: node.offsetHeight, width: node.offsetWidth }
  }, [captureNatural, vector])

  const apply = useCallback(
    (next: Point, nextScale: number) => {
      const clamped = clampPan(next, baseSize(), nextScale)
      offsetRef.current = clamped
      paint(clamped, nextScale)
      setScale(nextScale)
    },
    [baseSize, paint]
  )

  const reset = useCallback(() => {
    offsetRef.current = ORIGIN
    const node = imageRef.current
    if (node && vector) {
      // Hand the box back to the stylesheet; the fit classes take over again.
      node.style.width = ''
      node.style.height = ''
      node.style.maxWidth = ''
      node.style.maxHeight = ''
      node.style.transform = 'translate3d(0px, 0px, 0)'
    } else {
      paint(ORIGIN, MIN_ZOOM)
    }
    setScale(MIN_ZOOM)
  }, [paint, vector])

  /** Zoom from a control, anchored at the image's center. */
  const zoomByStep = useCallback(
    (factor: number) => {
      setScale(prev => {
        const next = clampZoom(prev * factor)
        const scaled = { x: (offsetRef.current.x * next) / prev, y: (offsetRef.current.y * next) / prev }
        const clamped = clampPan(scaled, baseSize(), next)
        offsetRef.current = clamped
        paint(clamped, next)
        return next
      })
    },
    [baseSize, paint]
  )

  const zoomIn = useCallback(() => zoomByStep(ZOOM_STEP), [zoomByStep])
  const zoomOut = useCallback(() => zoomByStep(1 / ZOOM_STEP), [zoomByStep])

  // Reopening must not inherit the last session's zoom.
  useEffect(() => {
    if (!active) reset()
  }, [active, reset])

  const onWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      event.preventDefault()
      const node = imageRef.current
      if (!node) return
      const rect = node.getBoundingClientRect()
      const cursor = {
        x: event.clientX - (rect.left + rect.width / 2),
        y: event.clientY - (rect.top + rect.height / 2)
      }
      const next = zoomFromWheel(scale, event.deltaY)
      const result = zoomAtPoint(cursor, offsetRef.current, scale, next, baseSize())
      offsetRef.current = result.offset
      paint(result.offset, result.scale)
      setScale(result.scale)
    },
    [baseSize, paint, scale]
  )

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLImageElement>) => {
      if (scale <= MIN_ZOOM || event.button !== 0) return
      event.preventDefault()
      dragRef.current = {
        origin: { ...offsetRef.current },
        pointerId: event.pointerId,
        start: { x: event.clientX, y: event.clientY }
      }
      event.currentTarget.setPointerCapture(event.pointerId)
    },
    [scale]
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLImageElement>) => {
      const drag = dragRef.current
      if (!drag || drag.pointerId !== event.pointerId) return
      const next = clampPan(
        {
          x: drag.origin.x + (event.clientX - drag.start.x),
          y: drag.origin.y + (event.clientY - drag.start.y)
        },
        baseSize(),
        scale
      )
      offsetRef.current = next
      paint(next, scale)
    },
    [baseSize, paint, scale]
  )

  const endDrag = useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    dragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  /** True while a drag is in flight, so the click handler can stand down. */
  const isPanning = useCallback(() => dragRef.current !== null, [])

  const onDoubleClick = useCallback(() => {
    if (scale > MIN_ZOOM) reset()
    else apply(ORIGIN, clampZoom(MIN_ZOOM * ZOOM_STEP * ZOOM_STEP))
  }, [apply, reset, scale])

  // Keyboard zoom, scoped so it never steals a character from the app.
  //
  // `+` and `-` are ordinary characters. A modal lightbox owns the screen and
  // may claim them; a pane sitting beside a live composer may NOT — there the
  // shortcut only applies while the pointer is over the image or the focus is
  // inside it. Either way, a typing target always wins.
  useEffect(() => {
    if (!active) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return
      // A modifier means the user is asking the OS/app for something else
      // (browser zoom, a chord) — not for this image.
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (!modal) {
        const container = containerRef.current
        if (!container) return
        const focusInside = container.contains(document.activeElement)
        if (!hoverRef.current && !focusInside) return
      }
      if (event.key === '+' || event.key === '=') {
        event.preventDefault()
        zoomIn()
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault()
        zoomOut()
      } else if (event.key === '0') {
        event.preventDefault()
        reset()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [active, modal, reset, zoomIn, zoomOut])

  return {
    canZoomIn: scale < MAX_ZOOM,
    canZoomOut: scale > MIN_ZOOM,
    containerProps: {
      onPointerEnter: () => {
        hoverRef.current = true
      },
      onPointerLeave: () => {
        hoverRef.current = false
      },
      ref: containerRef
    },
    endDrag,
    imageRef,
    isPanning,
    isZoomed: scale > MIN_ZOOM,
    onDoubleClick,
    onPointerDown,
    onPointerMove,
    onWheel,
    reset,
    scale,
    zoomIn,
    zoomOut
  }
}
