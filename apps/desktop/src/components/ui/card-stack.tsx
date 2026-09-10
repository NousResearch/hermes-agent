import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

interface CardStackProps {
  children: ReactNode
  count: number
  /** Match the existing surface: this primitive owns depth, never its palette. */
  backClassName: string
  direction?: 'up' | 'down'
}

/** One live card and at most two inert card edges, regardless of queue length. */
export function CardStack({ children, count, backClassName, direction = 'down' }: CardStackProps) {
  const depth = Math.min(2, Math.max(0, count - 1))

  return (
    <div
      className="relative isolate min-w-0"
      data-slot="card-stack"
      style={{ paddingTop: direction === 'up' ? depth * 6 : 0, paddingBottom: direction === 'down' ? depth * 6 : 0 }}
    >
      {Array.from({ length: depth }, (_, index) => {
        const layer = index + 1

        return (
          <div
            aria-hidden="true"
            className={cn('pointer-events-none absolute', backClassName)}
            data-slot="card-stack-edge"
            key={layer}
            style={{
              insetInline: layer * 6,
              top: direction === 'up' ? (depth - layer) * 6 : layer * 6,
              bottom: direction === 'up' ? layer * 6 : (depth - layer) * 6,
              zIndex: -layer
            }}
          />
        )
      })}
      <div className="relative">{children}</div>
    </div>
  )
}
