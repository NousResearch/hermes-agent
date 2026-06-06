import { forwardRef } from 'react'
import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

interface LogViewProps extends ComponentProps<'div'> {
  // Drop the box chrome (border + padding) when embedding inside a panel/card
  // that already supplies it — keeps the mono text style unified everywhere.
  bare?: boolean
}

// Shared raw-log / console-output viewer. One style for every place we surface
// logs (boot-failure recent logs, install output, command-center, …): no
// background, a hairline border, tight padding, small mono text. Pass a max-h-*
// via className per use.
export const LogView = forwardRef<HTMLDivElement, LogViewProps>(function LogView(
  { bare, className, ...props },
  ref
) {
  return (
    <div
      className={cn(
        'overflow-auto font-mono text-[0.6875rem] leading-[1.5] whitespace-pre-wrap break-words text-(--ui-text-tertiary) [scrollbar-width:thin]',
        !bare && 'rounded-lg border border-(--ui-stroke-tertiary) px-2.5 py-1.5',
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
