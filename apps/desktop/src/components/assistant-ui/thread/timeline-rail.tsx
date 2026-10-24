import './timeline.css'

import { useVirtualizer } from '@tanstack/react-virtual'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import {
  dockGrowth,
  dockRows,
  type TimelineBand,
  timelineBarWidth,
  type TimelineEntry,
  timelineIndexAt
} from './timeline-data'

interface TimelineRailProps {
  activeIndex: number
  entries: TimelineEntry[]
  loadingId: string | null
  onJump: (id: string) => void
}

/** Resting pitch in rem; the thread sizes the rail at this times the count. */
const PITCH_REM = 0.4375
/** The mark at rest, and how much more the ramp and the active prompt add. */
const BAR_REST_REM = 0.5
const BAR_GROW_REM = 1
/** How tall a fully magnified row becomes in rem: the label box's height. Rows
 *  are hit-tested as bands this tall, so a 7px pitch still yields a 24px target,
 *  and the label can sit at the row's scale. */
const DOCK_PEAK_REM = 1.5
/** The rail's own box grows in these steps as rows magnify. Quantised so the
 *  virtualizer re-ranges a few times per hover instead of on every frame. */
const PORT_STEP_PX = 8

/** Only the visible slice owns DOM. Pointer paint is local and never renders. */
export function TimelineRail({ activeIndex, entries, loadingId, onJump }: TimelineRailProps) {
  const root = useRef<HTMLDivElement>(null)
  const chip = useRef<HTMLSpanElement>(null)
  const hover = useRef<number | null>(null)
  const pointerY = useRef<number | null>(null)
  /**
   * The rail's scroll-port origin and scroll offset, cached. Reading either one
   * inside the pointer path forces a style/layout flush of the whole transcript,
   * and the first hover of a session pays that flush in full (the reported ~0.3s
   * stall). Both are refreshed where layout is already being measured anyway —
   * mount, ResizeObserver, scroll — so a pointer frame reads no layout at all.
   */
  const view = useRef({ origin: 0, scroll: 0 })
  /** The rows as last painted: hit-testing reads this, not the resting grid. */
  const bands = useRef<TimelineBand[]>([])
  /** The extra height currently given to the rail for the magnified rows. */
  const port = useRef(0)
  const frame = useRef(0)
  const focused = useRef(false)
  const pendingFocus = useRef<number | null>(null)
  const [rem, setRem] = useState(16)

  const pitch = rem * PITCH_REM

  const virtualizer = useVirtualizer({
    count: entries.length,
    estimateSize: () => pitch,
    getItemKey: index => entries[index].id,
    getScrollElement: () => root.current,
    initialRect: { height: 300, width: 48 },
    overscan: 4
  })

  const items = virtualizer.getVirtualItems()
  const visibleKey = `${items[0]?.index}:${items.at(-1)?.index}`

  useLayoutEffect(() => {
    const element = root.current

    if (!element) {
      return
    }

    const measure = () => {
      const size = parseFloat(getComputedStyle(element.ownerDocument.documentElement).fontSize)

      if (Number.isFinite(size) && size > 0) {
        setRem(size)
      }

      view.current.origin = element.getBoundingClientRect().top
      view.current.scroll = element.scrollTop
    }

    const observer = new ResizeObserver(measure)
    observer.observe(element)
    measure()
    // A pane can move without resizing (a toolbar appearing above it), so the
    // cached origin also follows window resizes.
    window.addEventListener('resize', measure)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [])

  useEffect(() => virtualizer.measure(), [pitch, virtualizer])

  const hoveredIndex = useCallback(() => {
    if (hover.current === null) {
      return null
    }

    return Math.max(0, Math.min(entries.length - 1, Math.round(hover.current)))
  }, [entries.length])

  /**
   * One rAF writes every per-pointer value: the marks' size and the label's
   * position/opacity. Rows never move, so React never re-renders on pointer
   * movement and no row can be displaced out of the rail.
   */
  const paint = useCallback(() => {
    frame.current = 0

    const scroller = root.current

    if (!scroller) {
      return
    }

    const anchor = pointerY.current === null ? null : pointerY.current - view.current.origin + view.current.scroll
    const ticks = Array.from(scroller.querySelectorAll<HTMLElement>('[data-slot="timeline-tick"]'))

    const rows = dockRows(
      ticks.map(tick => Number(tick.dataset.timelineIndex) * pitch),
      pitch,
      anchor,
      rem * (DOCK_PEAK_REM - PITCH_REM)
    )

    ticks.forEach((tick, position) => {
      const row = rows[position]
      const index = Number(tick.dataset.timelineIndex)
      const bar = tick.firstElementChild as HTMLElement | null

      tick.style.top = `${row.top}px`
      tick.style.height = `${row.height}px`

      if (!bar) {
        return
      }

      // The active prompt keeps a visible mark with the pointer elsewhere; the
      // Dock ramp stacks on top of it, so the cursor always wins.
      const width = Math.max(timelineBarWidth(index, activeIndex, null), BAR_REST_REM + BAR_GROW_REM * row.scale)

      bar.style.width = `${width}rem`
      bar.style.height = `${0.0625 + 0.25 * row.scale}rem`
    })

    bands.current = ticks.map((tick, position) => ({
      height: rows[position].height,
      index: Number(tick.dataset.timelineIndex),
      top: rows[position].top
    }))

    // The rail's own box has to grow by exactly what the rows added, or the
    // magnified rows are clipped by the resting height and simply disappear.
    const extra = anchor === null ? 0 : Math.ceil(dockGrowth(rows, pitch) / PORT_STEP_PX) * PORT_STEP_PX

    if (extra !== port.current) {
      port.current = extra
      scroller.style.height = extra > 0 ? `calc(100% + ${extra}px)` : ''
    }

    const label = chip.current

    if (!label) {
      return
    }

    const hovered = hoveredIndex()
    const first = ticks.length ? Number(ticks[0].dataset.timelineIndex) : 0
    const row = hovered === null ? undefined : rows[hovered - first]

    if (hovered === null || !row || !entries[hovered]) {
      label.style.opacity = '0'

      // The idle rail mounts no label text at all: the box is one node whose
      // content exists only while the pointer is on the rail.
      if (label.textContent) {
        label.textContent = ''
      }

      return
    }

    // Beside the row's own mark, centred on it: the mark stays visible.
    label.style.top = `${row.top + row.height / 2 - view.current.scroll}px`
    label.style.opacity = '1'

    const text = entries[hovered].label

    if (label.textContent !== text) {
      label.textContent = text
    }
  }, [activeIndex, entries, hoveredIndex, pitch, rem])

  const schedule = useCallback(() => {
    if (!frame.current) {
      frame.current = requestAnimationFrame(paint)
    }
  }, [paint])

  useLayoutEffect(() => {
    paint()

    if (pendingFocus.current !== null) {
      const button = root.current?.querySelector<HTMLButtonElement>(`[data-timeline-index="${pendingFocus.current}"]`)

      if (button) {
        button.focus({ preventScroll: true })
        pendingFocus.current = null
      }
    }

    return () => {
      cancelAnimationFrame(frame.current)
      frame.current = 0
    }
  }, [paint, visibleKey])

  useEffect(() => {
    if (hover.current === null && !focused.current) {
      virtualizer.scrollToIndex(activeIndex, { align: 'auto' })
    }
  }, [activeIndex, virtualizer])

  const updatePointer = useCallback(() => {
    if (pointerY.current === null) {
      return
    }

    const y = pointerY.current - view.current.origin + view.current.scroll

    // Painted bands, not the resting grid: the row under the cursor keeps the
    // point while the pointer is inside its band, so the box cannot flicker
    // between neighbours at the edges of a magnified row.
    hover.current = timelineIndexAt(y, bands.current, pitch, entries.length, hover.current)
    schedule()
  }, [entries.length, pitch, schedule])

  // Leaving is judged by the whole rail — the box hangs left of it and is part of
  // the same subtree, so reading the label does not dismiss it.
  return (
    <div
      className="thread-timeline-rail"
      data-slot="thread-timeline-rail"
      onPointerLeave={() => {
        pointerY.current = null
        hover.current = null
        schedule()
      }}
    >
      <div
        className="thread-timeline-ticks"
        data-slot="thread-timeline-ticks"
        onBlur={event => {
          if (!event.currentTarget.contains(event.relatedTarget)) {
            focused.current = false
            hover.current = null
            schedule()
          }
        }}
        onKeyDown={event => {
          const index = Number(
            (event.target as HTMLElement).closest<HTMLElement>('[data-timeline-index]')?.dataset.timelineIndex
          )

          const next = { ArrowUp: index - 1, ArrowDown: index + 1, Home: 0, End: entries.length - 1 }[event.key]

          if (next === undefined || !Number.isFinite(next)) {
            return
          }

          event.preventDefault()
          pendingFocus.current = Math.max(0, Math.min(entries.length - 1, next))
          virtualizer.scrollToIndex(pendingFocus.current, { align: 'auto' })
          root.current
            ?.querySelector<HTMLButtonElement>(`[data-timeline-index="${pendingFocus.current}"]`)
            ?.focus({ preventScroll: true })
        }}
        onPointerEnter={event => {
          // Entering can arrive without a move event (the pointer stopped on the
          // frame it crossed the edge), and the box is the click target — so the
          // entry itself paints.
          pointerY.current = event.clientY
          updatePointer()
        }}
        onPointerMove={event => {
          pointerY.current = event.clientY
          updatePointer()
        }}
        onScroll={event => {
          view.current.scroll = event.currentTarget.scrollTop
          updatePointer()
        }}
        ref={root}
      >
        <div className="thread-timeline-track" style={{ height: virtualizer.getTotalSize() }}>
          {items.map(item => {
            const entry = entries[item.index]

            return (
              <button
                aria-busy={loadingId === entry.id || undefined}
                aria-current={item.index === activeIndex ? 'location' : undefined}
                aria-label={entry.preview}
                className="thread-timeline-tick"
                data-slot="timeline-tick"
                data-timeline-id={entry.id}
                data-timeline-index={item.index}
                key={entry.id}
                onClick={() => onJump(entry.id)}
                onFocus={() => {
                  focused.current = true
                  hover.current = item.index
                  schedule()
                }}
                style={{ top: item.start, height: pitch }}
                type="button"
              >
                <span aria-hidden data-slot="timeline-bar" />
              </button>
            )
          })}
        </div>
      </div>
      {/* Outside the scroller: the label hangs left of the rail, which the
          vertical scroll port would otherwise clip. */}
      <span aria-hidden className="thread-timeline-dock" data-slot="thread-timeline-dock" ref={chip} />
    </div>
  )
}
