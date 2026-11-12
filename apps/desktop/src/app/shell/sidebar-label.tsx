import type * as React from 'react'

import { cn } from '@/lib/utils'

interface SidebarPanelLabelProps extends React.ComponentProps<'span'> {
  dotClassName?: string
  meta?: React.ReactNode
}

export function SidebarPanelLabel({ children, className, dotClassName, meta, ...props }: SidebarPanelLabelProps) {
  return (
    <span
      className={cn(
        'flex min-w-0 items-center gap-2 pl-2 text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-(--theme-primary)',
        className
      )}
      {...props}
    >
      <span aria-hidden="true" className={cn('dither inline-block size-2 shrink-0 rounded-[1px]', dotClassName)} />
      <span className="min-w-0 truncate leading-none">{children}</span>
      {meta && (
        <span className="shrink-0 text-[0.6875rem] font-medium tracking-normal text-(--ui-text-quaternary)">
          {meta}
        </span>
      )}
    </span>
  )
}
