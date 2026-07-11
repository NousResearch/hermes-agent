'use client'

import { type ComponentProps, useEffect, useRef, useState } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { useI18n } from '@/i18n'
import { ChevronLeftIcon, ChevronRightIcon, Pause, Play } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { mediaExternalUrl, mediaName, mediaSrc } from '@/lib/media'
import { openExternalLink } from '@/lib/external-link'

export interface MediaCarouselImage {
  src: string
  title?: string
}

interface MediaCarouselProps {
  images: MediaCarouselImage[]
  title?: string
  /** Autoplay interval in ms. When omitted, defaults to 1500ms. */
  intervalMs?: number
}

const DEFAULT_INTERVAL_MS = 1500
const MIN_INTERVAL_MS = 250

// SLIDE_DURATION must match the CSS transition duration used below so the
// crossfade timing stays in sync with the autoplay tick.
const SLIDE_DURATION_MS = 400

function useResolvedSrc(src: string): { src: string; failed: boolean } {
  const [resolved, setResolved] = useState<string>(src)
  const [failed, setFailed] = useState<boolean>(false)

  useEffect(() => {
    let cancelled = false

    setFailed(false)
    setResolved(src)

    mediaSrc(src)
      .then(value => {
        if (!cancelled) {
          setResolved(value)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true)
          setResolved(src)
        }
      })

    return () => {
      cancelled = true
    }
  }, [src])

  return { src: resolved, failed }
}

function CarouselFrame({
  image,
  className,
  ...props
}: { image: MediaCarouselImage } & ComponentProps<'img'>) {
  const { src } = useResolvedSrc(image.src)

  return (
    <ZoomableImage
      alt={image.title ?? mediaName(image.src)}
      className={cn('block max-h-[60vh] w-auto max-w-full rounded-lg object-contain', className)}
      containerClassName="my-0 flex min-h-0 w-full items-center justify-center"
      slot="aui_media-carousel-frame"
      src={src}
      {...props}
    />
  )
}

export function MediaCarousel({ images, title, intervalMs }: MediaCarouselProps) {
  const { t } = useI18n()
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const count = images.length
  const interval = Math.max(MIN_INTERVAL_MS, intervalMs && intervalMs > 0 ? intervalMs : DEFAULT_INTERVAL_MS)

  const go = (next: number) => setIndex(((next % count) + count) % count)
  const prev = () => go(index - 1)
  const next = () => go(index + 1)

  useEffect(() => {
    if (!playing || count <= 1) {
      return
    }

    const handle = window.setInterval(() => {
      setIndex(current => (current + 1) % count)
    }, interval)

    return () => window.clearInterval(handle)
  }, [playing, count, interval])

  // Keep the active index in range when the gallery contents change.
  useEffect(() => {
    if (index > count - 1) {
      setIndex(Math.max(0, count - 1))
    }
  }, [count, index])

  const current = images[index]

  return (
    <div
      className="my-2 block w-full max-w-2xl overflow-hidden rounded-xl border border-border bg-muted/35"
      data-slot="aui_media-carousel"
    >
      {title && (
        <div className="flex items-center gap-2 px-3 pt-2.5 text-xs font-medium text-muted-foreground">
          <span className="truncate">{title}</span>
        </div>
      )}

      <div className="flex items-center gap-2 px-3 pb-1 pt-1 text-xs text-muted-foreground">
        <span className="ml-auto tabular-nums opacity-70">
          {index + 1} / {count}
        </span>
      </div>

      <div className="relative flex items-center gap-1 px-2 py-2">
        {count > 1 && (
          <button
            aria-label={t.desktop.carouselPrevious}
            className="grid size-8 shrink-0 place-items-center rounded-full bg-background/70 text-foreground shadow-sm backdrop-blur transition-colors hover:bg-accent focus-visible:outline-none"
            onClick={prev}
            type="button"
          >
            <ChevronLeftIcon className="size-4" />
          </button>
        )}

        <div className="relative min-h-0 min-w-0 flex-1" data-slot="aui_media-carousel-stage">
          {images.map((image, slideIndex) => (
            <div
              className={cn(
                'flex min-h-0 w-full items-center justify-center transition-opacity ease-out',
                slideIndex === index ? 'opacity-100' : 'pointer-events-none absolute inset-0 opacity-0'
              )}
              key={image.src}
              style={{ transitionDuration: `${SLIDE_DURATION_MS}ms` }}
            >
              <CarouselFrame image={image} />
            </div>
          ))}
        </div>

        {count > 1 && (
          <button
            aria-label={t.desktop.carouselNext}
            className="grid size-8 shrink-0 place-items-center rounded-full bg-background/70 text-foreground shadow-sm backdrop-blur transition-colors hover:bg-accent focus-visible:outline-none"
            onClick={next}
            type="button"
          >
            <ChevronRightIcon className="size-4" />
          </button>
        )}
      </div>

      <div className="flex items-center gap-2 px-3 pb-2.5">
        {count > 1 && (
          <button
            aria-label={playing ? t.desktop.carouselPause : t.desktop.carouselPlay}
            className="grid size-7 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none"
            onClick={() => setPlaying(p => !p)}
            type="button"
          >
            {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </button>
        )}

        <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto">
          {images.map((image, thumbIndex) => (
            <button
              aria-label={image.title ?? mediaName(image.src)}
              className={cn(
                'h-9 w-14 shrink-0 overflow-hidden rounded-md border bg-black/10 transition-opacity',
                thumbIndex === index ? 'border-foreground/70 opacity-100' : 'border-transparent opacity-55 hover:opacity-90'
              )}
              key={image.src}
              onClick={() => go(thumbIndex)}
              type="button"
            >
              <CarouselFrame image={image} className="h-9 w-14 rounded-md object-cover" />
            </button>
          ))}
        </div>

        {count > 1 && (
          <button
            className="shrink-0 text-xs font-medium text-muted-foreground underline underline-offset-4 decoration-current/20 hover:text-foreground"
            onClick={() => openExternalLink(mediaExternalUrl(current.src))}
            type="button"
          >
            {t.desktop.carouselOpen}
          </button>
        )}
      </div>
    </div>
  )
}
