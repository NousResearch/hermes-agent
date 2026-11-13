import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

/** One row that scrolls sideways, snapping each child to the start edge.
 *  Children keep their own width; set it from the caller (`*:w-68`). */
export function Reel({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      {...props}
      className={cn(
        'flex snap-x snap-mandatory gap-3 overflow-x-auto overscroll-x-contain pb-2 *:shrink-0 *:snap-start',
        className
      )}
      data-slot="reel"
    />
  )
}
