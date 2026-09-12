import { type ReactNode, useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

/** Marquee line: glides its tail into view while `go` holds, snaps back
 *  after. Stays put when nothing overflows or reduced motion is preferred.
 *  The owner passes one shared `go` so a whole row scrolls together. */
export function HoverScroll({
  children,
  className,
  go,
  marqueeKey
}: {
  children: ReactNode
  className?: string
  go: boolean
  /** Test hook; never rendered. */
  marqueeKey?: string
}) {
  const outer = useRef<HTMLSpanElement>(null)
  const inner = useRef<HTMLSpanElement>(null)
  const [distance, setDistance] = useState(0)

  useEffect(() => {
    if (!go) {
      setDistance(0)

      return
    }

    const box = outer.current
    const content = inner.current

    if (!box || !content) {
      return
    }

    if (typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return
    }

    const overflow = content.scrollWidth - box.clientWidth

    if (overflow > 8) {
      setDistance(overflow)
    }
  }, [go, children])

  return (
    <span
      className={cn('block overflow-hidden', distance > 0 ? null : 'truncate', className)}
      data-marquee={marqueeKey ?? true}
      ref={outer}
    >
      <span
        className="inline-block whitespace-nowrap will-change-transform"
        ref={inner}
        style={
          distance > 0
            ? { transform: `translateX(${-distance}px)`, transition: `transform ${Math.max(1, distance / 32)}s linear` }
            : undefined
        }
      >
        {children}
      </span>
    </span>
  )
}
