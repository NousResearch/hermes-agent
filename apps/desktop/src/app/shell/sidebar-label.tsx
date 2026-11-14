import type * as React from 'react'

import { DitherDot } from '@/components/ui/dither-dot'
import { cn } from '@/lib/utils'

interface SidebarPanelLabelProps extends React.ComponentProps<'span'> {
  dotClassName?: string
  meta?: React.ReactNode
}

export function SidebarPanelLabel({ children, className, dotClassName, meta, ...props }: SidebarPanelLabelProps) {
  return (
    <span
      className={cn(
        // 0.7rem/0.1em rather than the 0.64rem/0.16em this started as: these
        // labels are often CJK ("已置顶", "会话"), where wide tracking only
        // spreads the strokes apart and 10px hanzi stop resolving. Latin still
        // reads as a tracked cap-height label at 11px.
        'flex min-w-0 items-center gap-2 pl-2 text-[0.7rem] font-semibold uppercase tracking-[0.1em] text-(--theme-primary)',
        className
      )}
      {...props}
    >
      <DitherDot className={dotClassName} />
      <span className="min-w-0 truncate leading-none">{children}</span>
      {meta && (
        <span className="shrink-0 text-[0.6875rem] font-medium tracking-normal text-(--ui-text-quaternary)">
          {meta}
        </span>
      )}
    </span>
  )
}
