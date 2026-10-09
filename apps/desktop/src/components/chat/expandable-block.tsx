'use client'

import { type ReactNode, useCallback, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface ExpandableBlockProps {
  children: ReactNode
  className?: string
}

export function ExpandableBlock({ children, className }: ExpandableBlockProps) {
  const innerRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState(false)
  const [overflowing, setOverflowing] = useState(false)
  const [horizontalOverflowing, setHorizontalOverflowing] = useState(false)
  const [horizontalAtEnd, setHorizontalAtEnd] = useState(false)

  // Measure inside ResizeObserver timing only (layout is clean there). A
  // synchronous mount-time scrollHeight read forces a reflow per instance,
  // and a tool-heavy transcript mounts dozens of these on a session switch.
  const measure = useCallback(() => {
    const el = innerRef.current

    if (el) {
      setOverflowing(el.scrollHeight > 121)
      const hasHorizontalOverflow = el.scrollWidth > el.clientWidth + 1

      setHorizontalOverflowing(hasHorizontalOverflow)
      setHorizontalAtEnd(!hasHorizontalOverflow || el.scrollLeft + el.clientWidth >= el.scrollWidth - 1)
    }
  }, [])

  const handleScroll = useCallback(() => {
    const el = innerRef.current

    if (!el || !horizontalOverflowing) {
      return
    }

    setHorizontalAtEnd(el.scrollLeft + el.clientWidth >= el.scrollWidth - 1)
  }, [horizontalOverflowing])

  useResizeObserver(measure, innerRef)

  return (
    <div className="relative">
      <div
        className={cn(
          // `scrollbar-overlay` opts out of the app-wide classic thin gutters so
          // this scroller keeps platform overlay bars (no always-on track).
          'scrollbar-overlay overflow-y-auto overflow-x-auto',
          expanded ? 'max-h-[40dvh]' : 'max-h-[7.5rem]',
          className
        )}
        data-horizontal-overflow={horizontalOverflowing ? '' : undefined}
        onScroll={handleScroll}
        ref={innerRef}
      >
        {children}
      </div>
      {horizontalOverflowing && !horizontalAtEnd && (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-5 bg-linear-to-l from-[var(--expandable-fade-from,var(--ui-chat-surface-background))] to-transparent"
          data-slot="expandable-horizontal-fade"
        />
      )}
      {overflowing && (
        // The fade is a pure overflow cue and must not intercept pointer events:
        // it spans the full bottom edge (over the horizontal scrollbar of a wide
        // code block AND the block's last line), so making it clickable killed
        // both sideways scrolling and text selection. Keep the fade
        // `pointer-events-none` and pin the only clickable target — a compact
        // toggle — to the right edge, clear of the draggable scrollbar track.
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex h-7 justify-end bg-linear-to-t from-[var(--expandable-fade-from,var(--ui-chat-surface-background))] to-transparent">
          <button
            aria-expanded={expanded}
            aria-label={expanded ? 'Collapse' : 'Expand'}
            className="pointer-events-auto flex h-7 w-9 cursor-pointer items-end justify-center pb-1 text-muted-foreground/70 transition-colors hover:text-foreground"
            onClick={() => setExpanded(v => !v)}
            type="button"
          >
            <ChevronDown className={cn('size-3.5 transition-transform', expanded && 'rotate-180')} />
          </button>
        </div>
      )}
    </div>
  )
}
