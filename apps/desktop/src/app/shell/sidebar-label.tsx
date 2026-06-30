import type * as React from 'react'

import { cn } from '@/lib/utils'

interface SidebarPanelLabelProps extends React.ComponentProps<'span'> {
  dotClassName?: string
}

export function SidebarPanelLabel({ children, className, dotClassName: _dotClassName, ...props }: SidebarPanelLabelProps) {
  // Claude-style group header: a plain, quiet gray caption (置顶 / 项目 / 对话) —
  // no uppercase, no wide tracking, no colored accent dot. Just a low-contrast
  // label that lets the rows beneath it carry the eye.
  return (
    <span
      className={cn(
        'flex min-w-0 items-center pl-2 text-[0.6875rem] font-semibold tracking-normal text-(--ui-text-quaternary)',
        className
      )}
      {...props}
    >
      <span className="min-w-0 truncate leading-none">{children}</span>
    </span>
  )
}
