import type { MouseEventHandler } from 'react'

import { cn } from '@/lib/utils'

interface ComposerDragRegionProps {
  dragging: boolean
  onDoubleClick: MouseEventHandler<HTMLDivElement>
}

export function ComposerDragRegion({ dragging, onDoubleClick }: ComposerDragRegionProps) {
  const cursor = dragging ? 'cursor-grabbing' : 'cursor-grab'

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 z-5"
      data-dragging={dragging ? '' : undefined}
      data-slot="composer-drag-region"
    >
      <div
        className={cn('pointer-events-auto absolute inset-x-0 top-0 h-3', cursor)}
        data-slot="composer-drag-hit-target-top"
        onDoubleClick={onDoubleClick}
      />
      <div
        className={cn('pointer-events-auto absolute inset-y-0 left-0 w-3', cursor)}
        data-slot="composer-drag-hit-target-left"
        onDoubleClick={onDoubleClick}
      />
      <div
        className={cn('pointer-events-auto absolute inset-y-0 right-0 w-3', cursor)}
        data-slot="composer-drag-hit-target-right"
        onDoubleClick={onDoubleClick}
      />
    </div>
  )
}
