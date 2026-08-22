'use client'

import { type ComponentProps, useCallback, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

interface SelectionRect {
  x: number
  y: number
  w: number
  h: number
}

export interface ArtifactImageSelectionProps extends Omit<ComponentProps<'div'>, 'children'> {
  alt?: string
  artifactId: string
  /** The image element to wrap. When omitted the component renders its own
   *  `<img>` with `crossOrigin="anonymous"`. */
  children?: React.ReactNode
  onImageLoad?: () => void
  src: string
}

/**
 * Wraps an `<img>` element with a drag-to-select marquee overlay and a
 * floating "Add to chat" button. The selected region is cropped via canvas,
 * converted to a PNG Blob, and handed to `addArtifactImageSelectionToChat`.
 *
 * Renders the `<img>` with `crossOrigin="anonymous"` so the canvas read is
 * never tainted by a CORS mismatch.
 */
export function ArtifactImageSelection({
  alt,
  artifactId,
  children,
  className,
  onImageLoad,
  src,
  ...props
}: ArtifactImageSelectionProps) {
  const { t } = useI18n()
  const wrapperRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [selection, setSelection] = useState<SelectionRect | null>(null)
  const [confirmed, setConfirmed] = useState<SelectionRect | null>(null)
  const [dragging, setDragging] = useState(false)
  const dragStart = useRef<{ x: number; y: number } | null>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Ignore right-click / modified clicks
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) {
      return
    }

    const rect = wrapperRef.current?.getBoundingClientRect()

    if (!rect) {
      return
    }

    setConfirmed(null)
    setDragging(true)
    dragStart.current = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    setSelection(null)
  }, [])

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging || !dragStart.current) {
        return
      }

      const rect = wrapperRef.current?.getBoundingClientRect()

      if (!rect) {
        return
      }

      const currentX = e.clientX - rect.left
      const currentY = e.clientY - rect.top
      const x = Math.min(dragStart.current.x, currentX)
      const y = Math.min(dragStart.current.y, currentY)
      const w = Math.abs(currentX - dragStart.current.x)
      const h = Math.abs(currentY - dragStart.current.y)

      if (w < 1 || h < 1) {
        setSelection(null)
      } else {
        setSelection({ x, y, w, h })
      }
    },
    [dragging]
  )

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging || !dragStart.current) {
        return
      }

      setDragging(false)

      // Capture the start point before nulling the ref so TypeScript
      // doesn't see a narrowed-to-never value.
      const start = dragStart.current

      dragStart.current = null

      // Finalize the current selection as confirmed
      const rect = wrapperRef.current?.getBoundingClientRect()

      if (!rect) {
        setSelection(null)

        return
      }

      const currentX = e.clientX - rect.left
      const currentY = e.clientY - rect.top
      const x = Math.min(start.x, currentX)
      const y = Math.min(start.y, currentY)
      const w = Math.abs(start.x - currentX)
      const h = Math.abs(start.y - currentY)

      if (w < 4 || h < 4) {
        // Too small — treat as a click-through, not a selection
        setSelection(null)
        setConfirmed(null)

        return
      }

      setSelection(null)
      setConfirmed({ x, y, w, h })
    },
    [dragging]
  )

  const handleAddToChat = useCallback(() => {
    const img = imgRef.current ?? wrapperRef.current?.querySelector('img')
    const rect = confirmed

    if (!img || !rect || rect.w < 1 || rect.h < 1) {
      return
    }

    // Determine the scale factor between the rendered image and its natural size
    const renderedW = img.clientWidth
    const renderedH = img.clientHeight

    if (!renderedW || !renderedH) {
      return
    }

    const scaleX = img.naturalWidth / renderedW
    const scaleY = img.naturalHeight / renderedH

    const sx = Math.round(rect.x * scaleX)
    const sy = Math.round(rect.y * scaleY)
    const sw = Math.round(rect.w * scaleX)
    const sh = Math.round(rect.h * scaleY)

    const canvas = document.createElement('canvas')

    canvas.width = sw
    canvas.height = sh

    const ctx = canvas.getContext('2d')

    if (!ctx) {
      return
    }

    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh)
    canvas.toBlob(
      blob => {
        if (!blob) {
          return
        }

        // Dynamically import the bridge to avoid circular deps
        void import('@/app/chat/composer/selection-composer-bridge').then(
          ({ addArtifactImageSelectionToChat }) => {
            void addArtifactImageSelectionToChat(blob, artifactId)
          }
        )
      },
      'image/png'
    )

    setConfirmed(null)
  }, [artifactId, confirmed])

  const clearSelection = useCallback(() => {
    setConfirmed(null)
    setSelection(null)
  }, [])

  return (
    <div
      className={cn('group/image-selection relative inline-block max-w-full select-none align-top', className)}
      data-slot="aui_artifact-image-selection"
      onMouseDown={handleMouseDown}
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      ref={wrapperRef}
      {...props}
    >
      {/* Render children or a default <img> */}
      {children ?? (
        <img
          alt={alt ?? ''}
          className="pointer-events-none block h-auto w-auto max-h-(--image-preview-height) max-w-full"
          crossOrigin="anonymous"
          onLoad={onImageLoad}
          ref={imgRef}
          src={src}
        />
      )}

      {/* Marquee rectangle during active drag */}
      {selection && (
        <svg
          aria-hidden
          className="pointer-events-none absolute inset-0 size-full"
        >
          <rect
            className="fill-blue-500/15 stroke-blue-500"
            height={selection.h}
            rx={2}
            strokeWidth={1.5}
            width={selection.w}
            x={selection.x}
            y={selection.y}
          />
        </svg>
      )}

      {/* Confirmed selection — show the dimmed overlay + "Add to chat" button */}
      {confirmed && (
        <>
          {/* Semi-transparent overlay */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 bg-black/40"
          />

          {/* Highlight the selected region */}
          <svg
            aria-hidden
            className="pointer-events-none absolute inset-0 size-full"
          >
            <rect
              className="fill-transparent stroke-blue-400"
              height={confirmed.h}
              rx={2}
              strokeWidth={2}
              width={confirmed.w}
              x={confirmed.x}
              y={confirmed.y}
            />
          </svg>

          {/* Floating "Add to chat" button */}
          <Tip label={t.composer.addImageToChat}>
            <button
              className="absolute z-10 grid size-8 place-items-center rounded-full border border-border/70 bg-background/90 text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-accent hover:text-foreground"
              onClick={handleAddToChat}
              style={{
                left: Math.max(4, confirmed.x + confirmed.w + 8),
                top: Math.max(4, confirmed.y + confirmed.h + 8)
              }}
              title={t.composer.addImageToChat}
              type="button"
            >
              <Codicon name="arrow-up" className="size-4" />
            </button>
          </Tip>

          {/* Dismiss button */}
          <Tip label={t.common?.cancel ?? 'Cancel'}>
            <button
              className="absolute z-10 grid size-6 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-accent hover:text-foreground"
              onClick={clearSelection}
              style={{
                left: Math.max(4, confirmed.x + confirmed.w + 8),
                top: Math.max(4, confirmed.y)
              }}
              title={t.common?.cancel ?? 'Cancel'}
              type="button"
            >
              <Codicon name="close" className="size-3.5" />
            </button>
          </Tip>
        </>
      )}
    </div>
  )
}