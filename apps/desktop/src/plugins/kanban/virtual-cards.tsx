/**
 * A column's cards, virtualized once there are enough of them to matter.
 *
 * The done column is a board's whole history, which on a long-lived board is thousands of rows.
 * Mounting all of them on every board change is what made the page crawl, so past VIRTUALIZE_AT rows
 * the column renders a scroll area sized for every card and mounts only the ones near the viewport
 * plus a small overscan. Nothing is capped or filtered out: every card is one scroll away.
 *
 * Rows are measured (`measureElement`), so a two-line title and a four-line summary both get the room
 * they paint. House pattern: `app/chat/sidebar/virtual-session-list.tsx`.
 *
 * Two behaviours keep a virtual list from feeling worse than a long one:
 *  - the card the reader is looking at holds its place when cards land above it (a completion lands at
 *    the top of done and, without this, pushes what they were reading down a row), and
 *  - the row that holds focus stays mounted, so scrolling never drops the keyboard's place.
 *
 * This module is the window only: the caller renders each slot's own card, so it never imports the
 * board's card component (no cycle with board.tsx).
 */

import { useVirtualizer } from '@tanstack/react-virtual'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

/** The 0.5rem (gap-2) the column puts between cards; the virtualizer lays rows out on its own. */
const ROW_GAP_PX = 8

/** Cold-start height of one card: a two-line title, a two-line summary and the footer row. Rows are
 *  measured on first paint, so this only has to be close enough that the scroll thumb does not jump. */
export const CARD_ROW_ESTIMATE_PX = 96

const OVERSCAN_ROWS = 8

/** Below this a column keeps the plain list: a virtualizer over twelve cards buys nothing and costs a
 *  resize observer, a sizer element and an absolutely placed row per card. */
export const VIRTUALIZE_AT = 40

/** One mounted row: where it goes, and what to attach its measurement to. */
export interface VirtualSlot {
  index: number
  item: { id: string }
  measure: (element: Element | null) => void
  style: { transform: string }
}

/** The scroll element, handed to the virtualizer once it exists. The first render has no element, so a
 *  callback ref is what lets the virtualizer see one. */
export function useScrollerElement(): [(element: HTMLDivElement | null) => void, HTMLElement | null] {
  const [element, setElement] = useState<HTMLDivElement | null>(null)
  const attach = useCallback((node: HTMLDivElement | null) => setElement(node), [])

  return [attach, element]
}

/**
 * The slots to mount for `items`, inside a scroll area `totalSize` tall. The caller renders each slot's
 * own card and gives it `ref={slot.measure}` and `style={slot.style}`.
 */
export function useCardWindow<T extends { id: string }>(
  items: T[],
  scroller: HTMLElement | null
): { slots: VirtualSlot[]; totalSize: number } {
  const virtualizer = useVirtualizer({
    count: items.length,
    estimateSize: () => CARD_ROW_ESTIMATE_PX,
    gap: ROW_GAP_PX,
    getItemKey: index => items[index]?.id ?? index,
    getScrollElement: () => scroller,
    // jsdom and the first frame have no measured rect yet; the real one takes over on observe.
    initialRect: { height: 800, width: 256 },
    // The browser's own offsetHeight: unlike a bounding rect it ignores the row's translateY.
    measureElement: element => (element as HTMLElement | null)?.offsetHeight ?? 0,
    overscan: OVERSCAN_ROWS,
    // The window, plus the row that holds focus: a scrolled-away focused row must not unmount, or focus
    // drops to the body and the keyboard user loses their place.
    rangeExtractor: range => {
      const indexes = new Set<number>()

      for (let i = range.startIndex - range.overscan; i <= range.endIndex + range.overscan; i += 1) {
        if (i >= 0 && i < items.length) {
          indexes.add(i)
        }
      }

      const focused = document.activeElement?.closest<HTMLElement>('[data-kanban-row]')?.dataset.index

      if (focused !== undefined && Number(focused) < items.length) {
        indexes.add(Number(focused))
      }

      return [...indexes].sort((a, b) => a - b)
    }
  })

  const slots = virtualizer.getVirtualItems().flatMap(item => {
    const entry = items[item.index]

    return entry
      ? [
          {
            index: item.index,
            item: entry,
            measure: virtualizer.measureElement,
            style: { transform: `translateY(${item.start}px)` }
          }
        ]
      : []
  })

  // ---- keep the reader's place when rows land above them --------------------

  // The anchor: the first row in view and its offset from the top of the viewport. Re-read on every
  // scroll, so it is always the card the reader is actually looking at.
  const anchor = useRef<null | { id: string; offset: number }>(null)
  // What the virtualizer last put in view, refreshed every render (scroll updates it, and a re-render
  // always follows a scroll-driven change).
  const inView = useRef<Array<{ index: number; start: number }>>([])

  inView.current = virtualizer.getVirtualItems()

  const capture = useCallback(() => {
    const first = inView.current[0]
    const id = first === undefined ? undefined : items[first.index]?.id

    if (first && id !== undefined && scroller) {
      anchor.current = { id, offset: first.start - scroller.scrollTop }
    }
  }, [items, scroller])

  useEffect(() => {
    if (!scroller) {
      return
    }

    scroller.addEventListener('scroll', capture, { passive: true })

    return () => scroller.removeEventListener('scroll', capture)
  }, [capture, scroller])

  const indexOf = useMemo(() => new Map(items.map((entry, index) => [entry.id, index])), [items])
  const starts = virtualizer.measurementsCache
  const measuredFor = useRef<null | unknown>(null)

  useLayoutEffect(() => {
    const held = anchor.current

    // First paint has nothing on screen to hold, and a render that did not change the rows holds trivially.
    if (measuredFor.current === items || !scroller) {
      measuredFor.current = items

      return
    }

    measuredFor.current = items

    if (!held) {
      return
    }

    const index = indexOf.get(held.id)
    const start = index === undefined ? undefined : starts[index]?.start

    if (start === undefined) {
      return
    }

    const wanted = start - held.offset

    if (Math.abs(wanted - scroller.scrollTop) >= 1) {
      scroller.scrollTop = wanted
    }
  }, [indexOf, items, scroller, starts])

  return { slots, totalSize: virtualizer.getTotalSize() }
}
