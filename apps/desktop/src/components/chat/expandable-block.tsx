'use client'

import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'

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

  useLayoutEffect(() => {
    const el = innerRef.current

    if (!el) {
      return
    }

    const measure = () => setOverflowing(el.scrollHeight > 121)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)

    return () => observer.disconnect()
  }, [])

  return (
    <div className="relative">
      <div className={cn('overflow-y-auto', expanded ? 'max-h-[40dvh]' : 'max-h-[7.5rem]', className)} ref={innerRef}>
        {children}
      </div>
      {overflowing && (
        <button
          aria-expanded={expanded}
          aria-label={expanded ? 'Collapse' : 'Expand'}
          className="absolute inset-x-0 bottom-0 flex h-7 cursor-pointer items-end justify-center bg-linear-to-t from-(--ui-chat-surface-background) to-transparent pb-1 text-muted-foreground/70 transition-colors hover:text-foreground"
          onClick={() => setExpanded(v => !v)}
          type="button"
        >
          <ChevronDown className={cn('size-3.5 transition-transform', expanded && 'rotate-180')} />
        </button>
      )}
    </div>
  )
}
