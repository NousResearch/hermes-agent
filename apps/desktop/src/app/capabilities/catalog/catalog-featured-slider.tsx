import { type ReactNode, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ChevronLeft, ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'

/** How long a slide holds before the hero advances on its own. */
const SLIDE_MS = 7000

const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

/** The featured hero as a slider once there is more than one pick: native
 *  scroll-snap slides (swipe/trackpad work as-is), prev/next and dots. It
 *  advances every SLIDE_MS unless the pointer or focus is inside, the window
 *  is hidden, or the user prefers reduced motion. One pick renders bare. */
export function CatalogFeaturedSlider({ slides }: { slides: ReactNode[] }) {
  const { t } = useI18n()
  const c = t.catalog
  const track = useRef<HTMLDivElement>(null)
  const [index, setIndex] = useState(0)
  const [held, setHeld] = useState(false)
  const count = slides.length

  const go = (next: number) => {
    const node = track.current

    if (!node) {
      return
    }

    const target = (next + count) % count
    node.scrollTo({ left: target * node.clientWidth, behavior: reducedMotion() ? 'auto' : 'smooth' })
    setIndex(target)
  }

  // Keep the dots honest when the user swipes or scrolls the track directly.
  const onScroll = () => {
    const node = track.current

    if (node && node.clientWidth) {
      setIndex(Math.round(node.scrollLeft / node.clientWidth))
    }
  }

  useEffect(() => {
    if (count < 2 || held || reducedMotion()) {
      return
    }

    const timer = window.setInterval(() => {
      if (!document.hidden) {
        go(index + 1)
      }
    }, SLIDE_MS)

    return () => window.clearInterval(timer)
  })

  if (count < 2) {
    return <>{slides[0]}</>
  }

  return (
    <div
      aria-label={c.featured}
      aria-roledescription="carousel"
      className="catalog-featured-slider"
      onBlur={() => setHeld(false)}
      onFocus={() => setHeld(true)}
      onPointerEnter={() => setHeld(true)}
      onPointerLeave={() => setHeld(false)}
      role="region"
    >
      <div className="catalog-featured-track" onScroll={onScroll} ref={track}>
        {slides.map((slide, slideIndex) => (
          <div
            aria-hidden={slideIndex !== index || undefined}
            aria-label={c.slideOf(slideIndex + 1, count)}
            aria-roledescription="slide"
            className="catalog-featured-slide"
            inert={slideIndex !== index || undefined}
            key={slideIndex}
            role="group"
          >
            {slide}
          </div>
        ))}
      </div>
      <div className="catalog-featured-controls">
        <Button aria-label={c.previousSlide} onClick={() => go(index - 1)} size="icon-xs" variant="secondary">
          <ChevronLeft />
        </Button>
        <div className="catalog-featured-dots">
          {slides.map((_, slideIndex) => (
            <button
              aria-current={slideIndex === index || undefined}
              aria-label={c.slideOf(slideIndex + 1, count)}
              className={cn('catalog-featured-dot', slideIndex === index && 'is-active')}
              key={slideIndex}
              onClick={() => go(slideIndex)}
              type="button"
            />
          ))}
        </div>
        <Button aria-label={c.nextSlide} onClick={() => go(index + 1)} size="icon-xs" variant="secondary">
          <ChevronRight />
        </Button>
      </div>
    </div>
  )
}
