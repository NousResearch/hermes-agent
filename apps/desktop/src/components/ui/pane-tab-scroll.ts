/**
 * Moving a horizontal tab strip that does not fit.
 *
 * `PaneTabStrip` hides its native scrollbar, so once the tabs overflow the
 * hidden ones had no visible affordance at all: a plain mouse wheel does
 * nothing over a horizontal-only scroller, and Shift+wheel / a two-finger
 * trackpad swipe is not discoverable. This module owns both missing pieces —
 * which physical ends have tabs clipped behind them (the arrow cues) and the
 * wheel-to-horizontal mapping — as helpers that can be tested without layout.
 *
 * The sibling `pane-shell/tree/renderer/tab-strip-scroll.ts` answers a
 * different question (keep the ACTIVE tab in view); this one is about a user
 * choosing to move the strip.
 */

import { type RefObject, useCallback, useLayoutEffect, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'

export interface TabStripEdges {
  /** Tabs are clipped behind the LEFT end of the strip. */
  left: boolean
  /** Tabs are clipped behind the RIGHT end of the strip. */
  right: boolean
}

const NO_EDGES: TabStripEdges = { left: false, right: false }

/** A scroller at rest can report a fractional offset; sub-pixel is not clipped. */
const EDGE_SLACK = 1

/** One wheel "line" when the event reports lines instead of pixels. */
const WHEEL_LINE = 16

/** `WheelEvent.deltaMode`: 0 pixels, 1 lines, 2 pages. */
const DELTA_LINE = 1
const DELTA_PAGE = 2

export interface TabStripGeometry {
  clientWidth: number
  /** Chromium's RTL scroller: 0 at the logical start, negative toward its end. */
  rtl: boolean
  scrollLeft: number
  scrollWidth: number
}

/** Which physical ends of the strip currently have tabs clipped behind them. */
export function tabStripEdges({ clientWidth, rtl, scrollLeft, scrollWidth }: TabStripGeometry): TabStripEdges {
  const max = scrollWidth - clientWidth

  if (max <= EDGE_SLACK) {
    return NO_EDGES
  }

  // Distance from the PHYSICAL left end. RTL offsets run from 0 down to -max,
  // so shifting by the full range reads the same way LTR does.
  const fromLeft = rtl ? max + scrollLeft : scrollLeft

  return { left: fromLeft > EDGE_SLACK, right: fromLeft < max - EDGE_SLACK }
}

/**
 * The horizontal distance one wheel event asks for, in reading order: positive
 * advances toward later tabs. 0 means the event is not ours — a horizontal
 * gesture already moves the strip, Ctrl/⌘ is zoom, or there is no motion.
 */
export function tabStripWheelDelta(
  event: Pick<WheelEvent, 'ctrlKey' | 'deltaMode' | 'deltaX' | 'deltaY' | 'metaKey'>,
  clientWidth: number
): number {
  if (event.ctrlKey || event.metaKey) {
    return 0
  }

  if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
    return 0
  }

  if (event.deltaMode === DELTA_LINE) {
    return event.deltaY * WHEEL_LINE
  }

  return event.deltaMode === DELTA_PAGE ? event.deltaY * clientWidth : event.deltaY
}

const isRtl = (el: HTMLElement) => getComputedStyle(el).direction === 'rtl'

/** One arrow press moves most of the visible width, so a long strip needs a
 *  couple of presses rather than one per tab. */
const scrollPage = (clientWidth: number) => Math.max(1, Math.round(clientWidth * 0.8))

const scrollBehavior = (): ScrollBehavior =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches ? 'auto' : 'smooth'

export interface TabStripScroll {
  /** Which physical ends have tabs clipped behind them — render a cue per end. */
  edges: TabStripEdges
  /** Move the strip one page toward a physical end. */
  scroll: (direction: 'left' | 'right') => void
}

/**
 * Track where `ref`'s scroller can still move and let a plain wheel move it.
 *
 * Measurement rides observers instead of every render: a tab added, removed or
 * re-labelled changes the content width without resizing the scroller, so a
 * ResizeObserver alone would miss it (MutationObserver covers that half) and a
 * layout read per render would sit on the strip's hot path.
 */
export function useTabStripScroll(ref: RefObject<HTMLElement | null>): TabStripScroll {
  const [edges, setEdges] = useState<TabStripEdges>(NO_EDGES)
  const edgesRef = useRef<TabStripEdges>(NO_EDGES)

  const measure = useCallback(() => {
    const el = ref.current

    if (!el) {
      return
    }

    const next = tabStripEdges({
      clientWidth: el.clientWidth,
      rtl: isRtl(el),
      scrollLeft: el.scrollLeft,
      scrollWidth: el.scrollWidth
    })

    // Same booleans → keep the state reference: a scroll through a long strip
    // must not re-render every tab on each tick.
    if (next.left !== edgesRef.current.left || next.right !== edgesRef.current.right) {
      edgesRef.current = next
      setEdges(next)
    }
  }, [ref])

  useResizeObserver(measure, ref)

  useLayoutEffect(() => {
    const el = ref.current

    if (!el) {
      return
    }

    el.addEventListener('scroll', measure, { passive: true })

    const mutations = typeof MutationObserver === 'undefined' ? null : new MutationObserver(measure)

    mutations?.observe(el, { attributes: true, characterData: true, childList: true, subtree: true })

    // Native + non-passive: React's delegated wheel listener is passive, and a
    // handled gesture must not also scroll whatever sits behind the strip.
    const onWheel = (event: WheelEvent) => {
      const delta = tabStripWheelDelta(event, el.clientWidth)

      if (!delta) {
        return
      }

      const rtl = isRtl(el)
      const step = rtl ? -delta : delta

      const ends = tabStripEdges({
        clientWidth: el.clientWidth,
        rtl,
        scrollLeft: el.scrollLeft,
        scrollWidth: el.scrollWidth
      })

      const room = step < 0 ? ends.left : ends.right

      if (!room) {
        return
      }

      el.scrollLeft += step
      event.preventDefault()
    }

    el.addEventListener('wheel', onWheel, { passive: false })

    return () => {
      el.removeEventListener('scroll', measure)
      el.removeEventListener('wheel', onWheel)
      mutations?.disconnect()
    }
  }, [measure, ref])

  const scroll = useCallback(
    (direction: 'left' | 'right') => {
      const el = ref.current

      if (!el) {
        return
      }

      const step = scrollPage(el.clientWidth)

      el.scrollBy({ behavior: scrollBehavior(), left: direction === 'left' ? -step : step })
    },
    [ref]
  )

  return { edges, scroll }
}
