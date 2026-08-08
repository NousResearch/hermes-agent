import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

import {
  ANNOTATION_COLORS,
  ANNOTATION_TOOLS,
  type AnnotationShape,
  type AnnotationTool,
  DEFAULT_ANNOTATION_COLOR,
  isClick,
  nextCalloutNumber,
  nextShapeId,
  type Point,
  shapeBounds
} from './annotation-model'

interface AnnotationCanvasProps {
  /** Image source (data URL or path) the user is annotating. */
  src: string
  onChange?: (shapes: AnnotationShape[]) => void
}

export interface AnnotationCanvasHandle {
  /** The underlying canvas element (for compositing on save). */
  canvas: HTMLCanvasElement | null
}

const STROKE_WIDTH = 3
const CALLOUT_RADIUS = 14

function drawArrowHead(ctx: CanvasRenderingContext2D, from: Point, to: Point, color: string) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x)
  const headLength = 14

  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(to.x, to.y)
  ctx.lineTo(to.x - headLength * Math.cos(angle - Math.PI / 6), to.y - headLength * Math.sin(angle - Math.PI / 6))
  ctx.lineTo(to.x - headLength * Math.cos(angle + Math.PI / 6), to.y - headLength * Math.sin(angle + Math.PI / 6))
  ctx.closePath()
  ctx.fill()
}

function drawShape(ctx: CanvasRenderingContext2D, shape: AnnotationShape) {
  ctx.save()
  ctx.strokeStyle = shape.color
  ctx.fillStyle = shape.color
  ctx.lineWidth = STROKE_WIDTH
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  const [start, end] = [shape.points[0], shape.points[shape.points.length - 1]]

  if (!start || !end) {
    ctx.restore()

    return
  }

  switch (shape.tool) {
    case 'rect': {
      const bounds = shapeBounds(shape)
      ctx.strokeRect(bounds.minX, bounds.minY, bounds.maxX - bounds.minX, bounds.maxY - bounds.minY)

      break
    }

    case 'ellipse': {
      const bounds = shapeBounds(shape)
      const width = bounds.maxX - bounds.minX
      const height = bounds.maxY - bounds.minY

      if (width === 0 && height === 0) {
        break
      }

      ctx.beginPath()
      ctx.ellipse(bounds.minX + width / 2, bounds.minY + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2)
      ctx.stroke()

      break
    }

    case 'arrow': {
      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.stroke()
      drawArrowHead(ctx, start, end, shape.color)

      break
    }

    case 'pen': {
      if (shape.points.length < 2) {
        break
      }

      ctx.beginPath()
      ctx.moveTo(shape.points[0]!.x, shape.points[0]!.y)

      for (const point of shape.points.slice(1)) {
        ctx.lineTo(point.x, point.y)
      }

      ctx.stroke()

      break
    }

    case 'callout': {
      const center = start
      ctx.beginPath()
      ctx.arc(center.x, center.y, CALLOUT_RADIUS, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(String(shape.number ?? ''), center.x, center.y + 0.5)

      break
    }
  }

  ctx.restore()
}

export const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, AnnotationCanvasProps>(
  function AnnotationCanvas({ src, onChange }: AnnotationCanvasProps, ref) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const imageRef = useRef<HTMLImageElement | null>(null)
    const [tool, setTool] = useState<AnnotationTool>('arrow')
    const [color, setColor] = useState(DEFAULT_ANNOTATION_COLOR)
    const [shapes, setShapes] = useState<AnnotationShape[]>([])
    const [history, setHistory] = useState<AnnotationShape[][]>([])
    const [redo, setRedo] = useState<AnnotationShape[][]>([])
    const inProgressRef = useRef<AnnotationShape | null>(null)
    const draggingRef = useRef(false)
    const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)

    const [drawScale, setDrawScale] = useState<{ scale: number; offsetX: number; offsetY: number }>({
      scale: 1,
      offsetX: 0,
      offsetY: 0
    })

    useImperativeHandle(ref, () => ({ canvas: canvasRef.current }), [])

    // Map a pointer event to canvas coordinates (canvas is the image at fit).
    const pointerToCanvas = useCallback((event: React.PointerEvent<HTMLCanvasElement>): Point => {
      const canvas = canvasRef.current

      if (!canvas) {
        return { x: 0, y: 0 }
      }

      const rect = canvas.getBoundingClientRect()
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height

      return { x, y }
    }, [])

    const commit = useCallback(
      (nextShapes: AnnotationShape[]) => {
        setShapes(nextShapes)
        onChange?.(nextShapes)
      },
      [onChange]
    )

    const pushHistory = useCallback(
      (before: AnnotationShape[]) => {
        setHistory(previous => [...previous.slice(-49), before])
        setRedo([])
      },
      []
    )

    const undo = useCallback(() => {
      if (history.length === 0) {
        return
      }

      const previous = history[history.length - 1]!
      setHistory(history.slice(0, -1))
      setRedo(previousRedo => [...previousRedo, shapes])
      commit(previous)
    }, [history, shapes, commit])

    const redoAction = useCallback(() => {
      if (redo.length === 0) {
        return
      }

      const next = redo[redo.length - 1]!
      setRedo(redo.slice(0, -1))
      setHistory(previousHistory => [...previousHistory, shapes])
      commit(next)
    }, [redo, shapes, commit])

    const clearAll = useCallback(() => {
      if (shapes.length === 0) {
        return
      }

      pushHistory(shapes)
      commit([])
    }, [shapes, pushHistory, commit])

    // Load the source image once.
    // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
    useEffect(() => {
      const image = new Image()
      imageRef.current = image

      image.onload = () => {
        setImageSize({ width: image.naturalWidth, height: image.naturalHeight })
      }

      image.src = src

      return () => {
        image.onload = null
      }
    }, [src])

    // Fit the image inside the canvas (up to a work area) preserving aspect.
    useEffect(() => {
      const canvas = canvasRef.current

      if (!canvas || !imageSize) {
        return
      }

      const MAX_W = 1280
      const MAX_H = 720
      const scale = Math.min(MAX_W / imageSize.width, MAX_H / imageSize.height, 1)
      const width = Math.max(1, Math.round(imageSize.width * scale))
      const height = Math.max(1, Math.round(imageSize.height * scale))
      canvas.width = width
      canvas.height = height
      setDrawScale({ scale, offsetX: 0, offsetY: 0 })
    }, [imageSize])

    // Redraw on committed state changes.
    useEffect(() => {
      const canvas = canvasRef.current
      const image = imageRef.current
      const ctx = canvas?.getContext('2d')

      if (!canvas || !ctx) {
        return
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (image && image.complete && image.naturalWidth > 0) {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
      }

      for (const shape of [...shapes, ...(inProgressRef.current ? [inProgressRef.current] : [])]) {
        drawShape(ctx, shape)
      }
    }, [shapes, imageSize, drawScale, tool, color])

    const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current) {
        return
      }

      event.preventDefault()
      canvasRef.current.setPointerCapture(event.pointerId)
      draggingRef.current = true
      const point = pointerToCanvas(event)

      if (tool === 'callout') {
        const shape: AnnotationShape = {
          id: nextShapeId(),
          tool: 'callout',
          color,
          points: [point],
          number: nextCalloutNumber(shapes)
        }

        pushHistory(shapes)
        commit([...shapes, shape])

        return
      }

      inProgressRef.current = {
        id: nextShapeId(),
        tool,
        color,
        points: [point]
      }
    }

    const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
      const inProgress = inProgressRef.current

      if (!draggingRef.current || !inProgress) {
        return
      }

      const point = pointerToCanvas(event)
      inProgressRef.current = { ...inProgress, points: [...inProgress.points, point] }
      // Live-draw the in-progress shape without re-rendering React state per
      // pointermove (keeps the overlay smooth on Retina screenshots).
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      const image = imageRef.current

      if (canvas && ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        if (image && image.complete && image.naturalWidth > 0) {
          ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
        }

        for (const shape of [...shapes, inProgressRef.current]) {
          drawShape(ctx, shape)
        }
      }
    }

    const handlePointerUp = () => {
      const inProgress = inProgressRef.current

      if (!draggingRef.current || !inProgress) {
        return
      }

      draggingRef.current = false
      inProgressRef.current = null

      // A click that didn't move is a no-op for drag tools — don't leave a
      // zero-size rect/arrow behind.
      if (isClick(inProgress.points) && inProgress.tool !== 'pen') {
        return
      }

      pushHistory(shapes)
      commit([...shapes, inProgress])
    }

    return (
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {ANNOTATION_TOOLS.map(toolId => (
            <button
              aria-pressed={tool === toolId}
              className={cn(
                'rounded-md border px-2 py-1 text-xs capitalize transition-colors',
                tool === toolId
                  ? 'border-primary bg-accent text-foreground'
                  : 'border-border/60 text-muted-foreground hover:bg-accent/40'
              )}
              key={toolId}
              onClick={() => setTool(toolId)}
              type="button"
            >
              {toolId}
            </button>
          ))}
          <span className="mx-1 h-4 w-px bg-border/60" />
          {ANNOTATION_COLORS.map(swatch => (
            <button
              aria-label={`Color ${swatch}`}
              aria-pressed={color === swatch}
              className={cn(
                'size-5 rounded-full border transition-transform hover:scale-110',
                color === swatch ? 'border-foreground ring-2 ring-foreground/30' : 'border-border/70'
              )}
              key={swatch}
              onClick={() => setColor(swatch)}
              style={{ backgroundColor: swatch }}
              type="button"
            />
          ))}
          <span className="mx-1 h-4 w-px bg-border/60" />
          <button
            className="rounded-md border border-border/60 px-2 py-1 text-xs text-muted-foreground hover:bg-accent/40 disabled:opacity-40"
            disabled={history.length === 0}
            onClick={undo}
            type="button"
          >
            Undo
          </button>
          <button
            className="rounded-md border border-border/60 px-2 py-1 text-xs text-muted-foreground hover:bg-accent/40 disabled:opacity-40"
            disabled={redo.length === 0}
            onClick={redoAction}
            type="button"
          >
            Redo
          </button>
          <button
            className="rounded-md border border-border/60 px-2 py-1 text-xs text-muted-foreground hover:bg-accent/40 disabled:opacity-40"
            disabled={shapes.length === 0}
            onClick={clearAll}
            type="button"
          >
            Clear
          </button>
        </div>
        <div className="overflow-hidden rounded-lg border border-border/60 bg-black/5">
          <canvas
            aria-label="Annotation canvas"
            className="block max-h-[70vh] max-w-full touch-none"
            data-slot="annotation-canvas"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            ref={canvasRef}
          />
        </div>
      </div>
    )
  }
)
