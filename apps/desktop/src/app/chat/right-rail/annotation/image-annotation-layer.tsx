import { useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { useI18n } from '@/i18n'
import { attachmentId } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { addComposerAttachment } from '@/store/composer'
import { notify, notifyError } from '@/store/notifications'

import {
  BRUSH_COLOR,
  BRUSH_WIDTH_PX,
  type BrushPoint,
  type BrushStroke,
  compositeAnnotatedImage,
  computeContainBox,
  type ContainBox,
  dataUrlToBytes,
  traceStroke
} from './image-annotate'

interface ImageAnnotationLayerProps {
  /** data URL of the image being previewed. */
  imageDataUrl: string
  /** File label used to name the exported attachment. */
  label: string
  /** Called when annotation mode ends (submit, cancel, or error). */
  onExit: () => void
}

type Phase = 'commenting' | 'drawing'

/**
 * Brush-annotation overlay for image previews: the user draws red strokes on
 * a canvas layered over the image, hits 完成, adds a comment, and the
 * composite (original + strokes at full resolution) lands in the composer as
 * an image attachment together with the comment text.
 */
export function ImageAnnotationLayer({ imageDataUrl, label, onExit }: ImageAnnotationLayerProps) {
  const { t } = useI18n()
  const copy = t.preview.web.annotation

  const [phase, setPhase] = useState<Phase>('drawing')
  const [strokes, setStrokes] = useState<BrushStroke[]>([])
  const [currentStroke, setCurrentStroke] = useState<BrushPoint[]>([])
  const [box, setBox] = useState<ContainBox | null>(null)
  const [composite, setComposite] = useState<string | null>(null)
  const [comment, setComment] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const wrapRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawingRef = useRef(false)

  // Measure the object-contain content box once the image (and its frame) is
  // known, and whenever the preview resizes.
  useEffect(() => {
    const measure = () => {
      const img = imgRef.current
      if (!img || !img.naturalWidth) {
        return
      }
      setBox(computeContainBox(img.clientWidth, img.clientHeight, img.naturalWidth, img.naturalHeight))
    }

    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [imageDataUrl])

  // Repaint the drawing canvas whenever strokes change.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !box) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = BRUSH_COLOR
    ctx.lineWidth = BRUSH_WIDTH_PX
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    const all = currentStroke.length > 0 ? [...strokes, { points: currentStroke }] : strokes
    for (const stroke of all) {
      traceStroke(ctx, stroke)
    }
  }, [strokes, currentStroke, box])

  // Escape: drawing → exit mode; commenting → back to drawing.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        if (phase === 'commenting') {
          setPhase('drawing')
          setComposite(null)
        } else {
          onExit()
        }
      }
    }

    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onExit, phase])

  const pointFromEvent = useCallback((event: React.MouseEvent): BrushPoint => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }, [])

  const onMouseDown = useCallback(
    (event: React.MouseEvent) => {
      if (event.button !== 0 || phase !== 'drawing') {
        return
      }
      drawingRef.current = true
      setCurrentStroke([pointFromEvent(event)])
    },
    [phase, pointFromEvent]
  )

  const onMouseMove = useCallback(
    (event: React.MouseEvent) => {
      if (!drawingRef.current) {
        return
      }
      const point = pointFromEvent(event)
      setCurrentStroke(prev => [...prev, point])
    },
    [pointFromEvent]
  )

  const endStroke = useCallback(() => {
    if (!drawingRef.current) {
      return
    }
    drawingRef.current = false
    setCurrentStroke(prev => {
      if (prev.length === 0) {
        return prev
      }
      setStrokes(strokesNow => [...strokesNow, { points: prev }])
      return []
    })
  }, [])

  const handleUndo = useCallback(() => {
    setStrokes(prev => prev.slice(0, -1))
  }, [])

  const handleClear = useCallback(() => {
    setStrokes([])
    setCurrentStroke([])
  }, [])

  const handleFinish = useCallback(async () => {
    if (!box || strokes.length === 0) {
      return
    }

    const url = await compositeAnnotatedImage(imageDataUrl, strokes, box)
    if (!url) {
      notifyError(new Error('composite failed'), copy.composeFailed)
      return
    }

    setComposite(url)
    setPhase('commenting')
  }, [box, copy.composeFailed, imageDataUrl, strokes])

  const handleSubmit = useCallback(async () => {
    if (!composite || submitting) {
      return
    }
    setSubmitting(true)

    try {
      const bytes = await dataUrlToBytes(composite)
      const savedPath = bytes ? await window.hermesDesktop?.saveImageBuffer(bytes, '.png') : ''

      if (!savedPath) {
        notify({ kind: 'error', message: copy.imageSaveFailed })
        return
      }

      const attachmentLabel = `${label.replace(/\.[^.]+$/, '')}-标注.png`
      addComposerAttachment({
        detail: savedPath,
        id: attachmentId('image', savedPath),
        kind: 'image',
        label: attachmentLabel,
        path: savedPath,
        previewUrl: composite
      })

      const commentText = comment.trim() || copy.noComment
      requestComposerInsert(`[图片标注] ${label}\n> ${commentText}`, { mode: 'block', target: 'main' })
      requestComposerFocus('main')
      notify({ kind: 'success', message: copy.sentToComposer })
      onExit()
    } catch (error) {
      notifyError(error, copy.imageSaveFailed)
    } finally {
      setSubmitting(false)
    }
  }, [comment, composite, copy.noComment, copy.sentToComposer, copy.imageSaveFailed, label, onExit, submitting])

  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center overflow-auto bg-transparent p-4">
      <div className="relative" ref={wrapRef}>
        <img
          alt={label}
          className="max-h-full max-w-full select-none rounded-lg object-contain shadow-sm"
          draggable={false}
          onLoad={() => {
            const img = imgRef.current
            if (img && img.naturalWidth) {
              setBox(computeContainBox(img.clientWidth, img.clientHeight, img.naturalWidth, img.naturalHeight))
            }
          }}
          ref={imgRef}
          src={imageDataUrl}
        />

        {box && box.width > 0 && phase === 'drawing' && (
          <canvas
            className="absolute cursor-crosshair"
            height={Math.round(box.height)}
            onMouseDown={onMouseDown}
            onMouseLeave={endStroke}
            onMouseMove={onMouseMove}
            onMouseUp={endStroke}
            ref={canvasRef}
            style={{ left: box.offsetX, top: box.offsetY }}
            width={Math.round(box.width)}
          />
        )}
      </div>

      {phase === 'drawing' ? (
        <div className="pointer-events-none absolute inset-x-0 top-3 flex justify-center">
          <div
            className={cn(
              'pointer-events-auto flex items-center gap-3 rounded-full py-1.5 pl-4 pr-1.5',
              'border border-white/10 bg-neutral-900/85 text-neutral-100 shadow-lg shadow-black/30',
              'backdrop-blur-md backdrop-saturate-150',
              'animate-in fade-in slide-in-from-top-2 duration-200'
            )}
            data-slot="image-annotation-toolbar"
          >
            <span className="flex items-center gap-2 text-xs font-medium">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />
              {copy.brushTitle(strokes.length)}
            </span>

            <div className="flex items-center gap-1">
              <button
                className="rounded-full px-3 py-1 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={strokes.length === 0}
                onClick={handleUndo}
                type="button"
              >
                {copy.undo}
              </button>
              <button
                className="rounded-full px-3 py-1 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={strokes.length === 0}
                onClick={handleClear}
                type="button"
              >
                {copy.clear}
              </button>
              <button
                className="rounded-full px-3 py-1 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200"
                onClick={onExit}
                type="button"
              >
                {copy.cancelSession}
              </button>
              <button
                className="rounded-full bg-red-500 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={strokes.length === 0}
                onClick={() => void handleFinish()}
                type="button"
              >
                {copy.finishSession}
              </button>
            </div>
          </div>
        </div>
      ) : (
        composite && (
          <div
            className={cn(
              'pointer-events-auto absolute bottom-4 left-1/2 w-[24rem] -translate-x-1/2',
              'rounded-2xl border border-white/10 bg-neutral-900/90 shadow-2xl shadow-black/40',
              'backdrop-blur-xl backdrop-saturate-150',
              'animate-in fade-in slide-in-from-bottom-3 zoom-in-95 duration-200'
            )}
            data-testid="image-annotation-comment"
          >
            <div className="flex items-center gap-2.5 px-4 pb-2 pt-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-500 text-xs font-semibold text-white">
                🖊
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-neutral-100">{copy.brushCommentTitle}</div>
                <div className="truncate font-mono text-[0.6875rem] text-neutral-400">{label}</div>
              </div>
              <img
                alt={copy.screenshotAlt}
                className="h-10 w-16 shrink-0 rounded-md border border-white/10 object-cover"
                src={composite}
              />
            </div>

            <div className="px-4 pb-3">
              <textarea
                autoFocus
                className={cn(
                  'min-h-20 w-full resize-none rounded-xl border border-white/10 bg-white/5 px-3 py-2',
                  'text-sm text-neutral-100 placeholder:text-neutral-500',
                  'outline-none transition-colors focus:border-red-500/50 focus:bg-white/[0.07]'
                )}
                onChange={event => setComment(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                    event.stopPropagation()
                    void handleSubmit()
                  }
                }}
                placeholder={copy.placeholder}
                value={comment}
              />
              <div className="mt-2.5 flex items-center justify-between">
                <span className="text-[0.6875rem] text-neutral-500">{copy.hint}</span>
                <div className="flex gap-2">
                  <button
                    className="rounded-full px-3 py-1.5 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200"
                    onClick={() => {
                      setPhase('drawing')
                      setComposite(null)
                    }}
                    type="button"
                  >
                    {copy.back}
                  </button>
                  <button
                    className="rounded-full bg-red-500 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-red-400 disabled:opacity-50"
                    disabled={submitting}
                    onClick={() => void handleSubmit()}
                    type="button"
                  >
                    {copy.submit}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )
      )}
    </div>
  )
}
