import type * as React from 'react'

import { cn } from '@/lib/utils'

interface SidebarPanelLabelProps extends React.ComponentProps<'span'> {
  dotClassName?: string
  /** Replaces the dither dot inside the leading slot (e.g. a section's identity
   *  icon). The slot keeps the label's left edge stable either way. */
  glyph?: React.ReactNode
  /** Sizes the leading slot. The chat sidebar passes `w-3.5` so the dot/glyph
   *  centers on the same column as the session-row status dots; without it the
   *  slot hugs the dot and the label renders exactly as before. */
  slotClassName?: string
}

export function SidebarPanelLabel({
  children,
  className,
  dotClassName,
  glyph,
  slotClassName,
  ...props
}: SidebarPanelLabelProps) {
  return (
    <span
      className={cn(
        'flex min-w-0 items-center gap-2 pl-2 text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-(--theme-primary)',
        className
      )}
      {...props}
    >
      <span aria-hidden="true" className={cn('grid shrink-0 place-items-center', slotClassName)}>
        {glyph ?? <span className={cn('dither inline-block size-2 shrink-0 rounded-[1px]', dotClassName)} />}
      </span>
      <span className="min-w-0 truncate leading-none">{children}</span>
    </span>
  )
}
