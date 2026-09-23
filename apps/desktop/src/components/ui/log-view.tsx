import { forwardRef, type ComponentProps } from 'react'

import { cn } from '@/lib/utils'

// Shared raw-log viewer: no bg, hairline border, tight padding, small mono.
// One style everywhere we surface logs. Pass a max-h-* via className.
// Selectable by default — logs exist to be read and copied.
export const LogView = forwardRef<HTMLDivElement, ComponentProps<'div'>>(function LogView({ className, ...props }, ref) {
  return (
    <div
      ref={ref}
      className={cn(
        'overflow-auto rounded-lg border border-(--ui-stroke-tertiary) px-2.5 py-1.5 font-mono text-[0.6875rem] leading-[1.5] whitespace-pre-wrap break-words text-(--ui-text-tertiary)',
        className
      )}
      data-selectable-text="true"
      {...props}
    />
  )
})
