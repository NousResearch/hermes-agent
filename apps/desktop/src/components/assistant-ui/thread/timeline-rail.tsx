import './timeline.css'

import { useVirtualizer } from '@tanstack/react-virtual'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { Tip } from '@/components/ui/tooltip'

import { timelineBarWidth, type TimelineEntry } from './timeline-data'

interface TimelineRailProps {
  activeIndex: number
  entries: TimelineEntry[]
  loadingId: string | null
  onJump: (id: string) => void
}

/** Only the visible slice owns DOM or tooltip instances. Pointer paint is local. */
export function TimelineRail({ activeIndex, entries, loadingId, onJump }: TimelineRailProps) {
  const root = useRef<HTMLDivElement>(null)
  const hover = useRef<number | null>(null)
  const pointerY = useRef<number | null>(null)
  const frame = useRef(0)
  const focused = useRef(false)
  const pendingFocus = useRef<number | null>(null)
  const [pitch, setPitch] = useState(7)

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
      const rem = parseFloat(getComputedStyle(element.ownerDocument.documentElement).fontSize)

      if (Number.isFinite(rem) && rem > 0) {
        setPitch(rem * 0.4375)
      }
    }

    const observer = new ResizeObserver(measure)
    observer.observe(element)
    measure()

    return () => observer.disconnect()
  }, [])

  useEffect(() => virtualizer.measure(), [pitch, virtualizer])

  const paint = useCallback(() => {
    frame.current = 0

    for (const bar of root.current?.querySelectorAll<HTMLElement>('[data-slot="timeline-bar"]') ?? []) {
      const width =
        pointerY.current === null
          ? '0.5rem'
          : `${Number(timelineBarWidth(Number(bar.dataset.index), activeIndex, hover.current).toFixed(4))}rem`

      if (bar.style.width !== width) {
        bar.style.width = width
      }
    }
  }, [activeIndex])

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
    const scroller = root.current

    if (!scroller || pointerY.current === null) {
      return
    }

    const y = pointerY.current - scroller.getBoundingClientRect().top + scroller.scrollTop
    hover.current = Math.max(0, Math.min(entries.length - 1, y / pitch - 0.5))
    schedule()
  }, [entries.length, pitch, schedule])

  return (
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
      onPointerLeave={() => {
        pointerY.current = null
        hover.current = null
        schedule()
      }}
      onPointerMove={event => {
        pointerY.current = event.clientY
        updatePointer()
      }}
      onScroll={updatePointer}
      ref={root}
    >
      <div className="thread-timeline-track" style={{ height: virtualizer.getTotalSize() }}>
        {items.map(item => {
          const entry = entries[item.index]

          return (
            <Tip
              key={entry.id}
              label={entry.preview}
              placement="right-rail"
              sideOffset={8}
              style={{ textAlign: 'right' }}
            >
              <button
                aria-busy={loadingId === entry.id || undefined}
                aria-current={item.index === activeIndex ? 'location' : undefined}
                aria-label={entry.preview}
                className="thread-timeline-tick"
                data-timeline-id={entry.id}
                data-timeline-index={item.index}
                onClick={() => onJump(entry.id)}
                onFocus={() => {
                  focused.current = true
                  hover.current = item.index
                  schedule()
                }}
                style={{ top: item.start, height: pitch }}
                type="button"
              >
                <span aria-hidden data-index={item.index} data-slot="timeline-bar" />
              </button>
            </Tip>
          )
        })}
      </div>
    </div>
  )
}
