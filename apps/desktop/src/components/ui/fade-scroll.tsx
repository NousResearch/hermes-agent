import { type CSSProperties, type ReactNode, useLayoutEffect, useRef } from 'react'

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
export function edgeMask({ above, below }: FadeEdges, axis: 'x' | 'y' = 'y'): string | undefined {
  if (!above && !below) {
    return undefined
  }

  const top = above ? `transparent, black ${FADE}` : 'black'
  const bottom = below ? `black calc(100% - ${FADE}), transparent` : 'black'

  return `linear-gradient(to ${axis === 'x' ? 'right' : 'bottom'}, ${top}, ${bottom})`
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
 * The gradient is EDGE-AWARE and pure CSS: the `scroll-fade-y` utility
 * (styles.css) drives the mask off the scroller's own scroll position with
 * scroll-driven animations. A side only fades while content is clipped behind
 * it, a list that fits shows no gradient at all, and a list scrolled to the
 * bottom stops fading its last row, with no scroll or resize listener and no
 * React state. A browser without scroll-driven animations (Chromium < 115)
 * shows no fade rather than a broken one.
 *
 * From each edge inward the mask is `pad` of fully hidden content, then `fade`
 * of gradient. Set `pad` to the scroller's own padding so the gradient starts
 * at the content instead of spending itself on the empty padding. `deps`
 * re-pins the scroller to the bottom when it changes — newest-at-bottom feeds
 * want that; a plain list should leave it unset.
 */
export function FadeScroll({
  children,
  className,
  deps,
  fade,
  maxHeight = '9rem',
  pad
}: {
  children: ReactNode
  className?: string
  deps?: unknown
  fade?: string
  maxHeight?: string
  pad?: string
}) {
  const ref = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    if (deps !== undefined && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }
  }, [deps])

  const style = {
    maxHeight,
    ...(fade ? { '--scroll-fade-size': fade } : {}),
    ...(pad ? { '--scroll-fade-pad': pad } : {})
  } as CSSProperties

  return (
    <div className={cn('scroll-fade-y overflow-y-auto overscroll-contain', className)} ref={ref} style={style}>
      {children}
    </div>
  )
}
