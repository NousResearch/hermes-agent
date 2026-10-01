'use client'

import { type ComponentProps, useEffect, useState } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { useI18n } from '@/i18n'
import { ChevronLeftIcon, ChevronRightIcon, Pause, Play } from '@/lib/icons'
import { mediaName, resolveMediaDisplaySrc } from '@/lib/media'
import { cn } from '@/lib/utils'

export interface MediaCarouselImage {
  src: string
  title?: string
}

interface MediaCarouselProps {
  images: MediaCarouselImage[]
  title?: string
  /** Autoplay interval in ms. When omitted, defaults to 4000ms. */
  intervalMs?: number
}

const DEFAULT_INTERVAL_MS = 4000
const MIN_INTERVAL_MS = 500

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

    resolveMediaDisplaySrc(src)
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

function CarouselFrame({ image, className, ...props }: { image: MediaCarouselImage } & ComponentProps<'img'>) {
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

  // Chrome is hover-only — the image is the content, a panel around it is noise.
  // `focus-visible` keeps every control reachable by keyboard.
  const hoverControl =
    'absolute grid place-items-center rounded-full bg-background/70 text-foreground opacity-0 shadow-sm backdrop-blur transition-opacity focus-visible:opacity-100 group-hover/carousel:opacity-100 hover:bg-accent'

  return (
    <div className="group/carousel relative my-2 w-full max-w-2xl" data-slot="aui_media-carousel">
      {title && <div className="mb-1.5 text-xs font-medium text-muted-foreground">{title}</div>}

      <div className="relative" data-slot="aui_media-carousel-stage">
        {images.map((image, slideIndex) => (
          <div
            className={cn(
              'flex w-full items-center justify-center transition-opacity ease-out',
              slideIndex === index ? 'opacity-100' : 'pointer-events-none absolute inset-0 opacity-0'
            )}
            key={image.src}
            style={{ transitionDuration: `${SLIDE_DURATION_MS}ms` }}
          >
            <CarouselFrame image={image} />
          </div>
        ))}

        {count > 1 && (
          <>
            <button
              aria-label={t.desktop.carouselPrevious}
              className={cn(hoverControl, 'left-1 top-1/2 size-8 -translate-y-1/2')}
              onClick={prev}
              type="button"
            >
              <ChevronLeftIcon className="size-4" />
            </button>
            <button
              aria-label={t.desktop.carouselNext}
              className={cn(hoverControl, 'right-1 top-1/2 size-8 -translate-y-1/2')}
              onClick={next}
              type="button"
            >
              <ChevronRightIcon className="size-4" />
            </button>
            <button
              aria-label={playing ? t.desktop.carouselPause : t.desktop.carouselPlay}
              className={cn(hoverControl, 'bottom-1.5 right-1.5 size-7')}
              onClick={() => setPlaying(p => !p)}
              type="button"
            >
              {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            </button>
          </>
        )}
      </div>

      {count > 1 && (
        <div className="mt-2.5 flex items-center justify-center gap-1.5" data-slot="aui_media-carousel-dots">
          {images.map((image, dotIndex) => (
            <button
              aria-current={dotIndex === index ? 'true' : undefined}
              aria-label={image.title ?? mediaName(image.src)}
              className={cn(
                'h-1.5 rounded-full transition-all',
                dotIndex === index ? 'w-4 bg-foreground/80' : 'w-1.5 bg-foreground/25 hover:bg-foreground/50'
              )}
              data-slot="aui_media-carousel-dot"
              key={image.src}
              onClick={() => go(dotIndex)}
              type="button"
            />
          ))}
        </div>
      )}
    </div>
  )
}
