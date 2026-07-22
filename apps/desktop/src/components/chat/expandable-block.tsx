'use client'

import { type ReactNode, useCallback, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface ExpandableBlockProps {
  children: ReactNode
  className?: string
  // `full` (default) spans the collapse button across the whole bottom edge —
  // used for plain text fallbacks. `end` pins it to the right edge so it never
  // covers a horizontal scrollbar living along the bottom of the inner
  // content (e.g. wide code blocks), leaving that scrollbar fully draggable.
  collapseAlign?: 'full' | 'end'
}

export function ExpandableBlock({
  children,
  className,
  collapseAlign = 'full'
}: ExpandableBlockProps) {
  const innerRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState(false)
  const [overflowing, setOverflowing] = useState(false)

  // Measure inside ResizeObserver timing only (layout is clean there). A
  // synchronous mount-time scrollHeight read forces a reflow per instance,
  // and a tool-heavy transcript mounts dozens of these on a session switch.
  const measure = useCallback(() => {
    const el = innerRef.current

    if (el) {
      setOverflowing(el.scrollHeight > 121)
    }
  }, [])

  useResizeObserver(measure, innerRef)

  // `end` alignment needs horizontal scroll room so the scrollbar below the
  // content stays clear of the button; pad the right edge to seat the button.
  const innerClassName = cn(
    'overflow-y-auto overflow-x-auto',
    collapseAlign === 'end' && 'pr-2',
    expanded ? 'max-h-[40dvh]' : 'max-h-[7.5rem]',
    className
  )

  const buttonClassName = cn(
    'absolute bottom-0 flex h-7 cursor-pointer items-end bg-linear-to-t from-(--ui-chat-surface-background) to-transparent pb-1 text-muted-foreground/70 transition-colors hover:text-foreground',
    collapseAlign === 'end' ? 'right-0 justify-end pr-2' : 'inset-x-0 justify-center'
  )

  return (
    <div className="relative">
      <div className={innerClassName} ref={innerRef}>
        {children}
      </div>
      {overflowing && (
        <button
          aria-expanded={expanded}
          aria-label={expanded ? 'Collapse' : 'Expand'}
          className={buttonClassName}
          onClick={() => setExpanded(v => !v)}
          type="button"
        >
          <ChevronDown className={cn('size-3.5 transition-transform', expanded && 'rotate-180')} />
        </button>
      )}
    </div>
  )
}
