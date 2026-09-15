import { type ReactNode, useCallback, useLayoutEffect, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { cn } from '@/lib/utils'

/** How much of a clipped edge the gradient eats. */
const FADE = '1.25rem'

export interface FadeEdges {
  above: boolean
  below: boolean
}

/**
 * The mask for a pair of clipped edges, or `undefined` when nothing is clipped
 * — a list that fits must not be dimmed at all.
 */
export function edgeMask({ above, below }: FadeEdges): string | undefined {
  if (!above && !below) {
    return undefined
  }

  const top = above ? `transparent, black ${FADE}` : 'black'
  const bottom = below ? `black calc(100% - ${FADE}), transparent` : 'black'

  return `linear-gradient(to bottom, ${top}, ${bottom})`
}

/** Which edges of a scroller currently have content clipped behind them. */
export function scrollEdges(el: Pick<HTMLElement, 'clientHeight' | 'scrollHeight' | 'scrollTop'>): FadeEdges {
  return {
    above: el.scrollTop > 1,
    below: el.scrollTop + el.clientHeight < el.scrollHeight - 1
  }
}

/**
 * A height-capped scroller whose clipped edges fade out.
 *
 * The fade is a mask-image, not an overlay: it resolves against whatever
 * background the parent happens to have, so one component works on the chat
 * backdrop, inside a widget panel, and in a drawer without knowing any of
 * their fills. Same technique as FadeText, on the other axis.
 *
 * The gradient is EDGE-AWARE — a side only fades when it actually has content
 * clipped behind it, so a list that fits shows no gradient at all and a list
 * scrolled to the bottom stops fading its last row. That state is tracked on
 * scroll and on resize (via the app's shared observer, so N of these cost one
 * delivery per frame rather than N).
 *
 * `deps` re-pins the scroller to the bottom when it changes — newest-at-bottom
 * feeds want that; a plain list should leave it unset.
 */
export function FadeScroll({
  children,
  className,
  deps,
  maxHeight = '9rem'
}: {
  children: ReactNode
  className?: string
  deps?: unknown
  maxHeight?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [edges, setEdges] = useState({ above: false, below: false })

  const measure = useCallback(() => {
    const el = ref.current

    if (!el) {
      return
    }

    const next = scrollEdges(el)

    setEdges(prev => (prev.above === next.above && prev.below === next.below ? prev : next))
  }, [])

  useLayoutEffect(() => {
    if (deps !== undefined && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }

    measure()
  }, [deps, measure])

  useResizeObserver(measure, ref)

  const mask = edgeMask(edges)

  return (
    <div
      className={cn('overflow-y-auto overscroll-contain', className)}
      onScroll={measure}
      ref={ref}
      style={mask ? { maskImage: mask, maxHeight, WebkitMaskImage: mask } : { maxHeight }}
    >
      {children}
    </div>
  )
}
