import type { ComponentProps } from 'react'
import { memo } from 'react'

import { cn } from '@/lib/utils'

export type StatusTone = 'good' | 'muted' | 'warn' | 'bad'

const TONE_STYLE: Record<StatusTone, string> = {
  // Shape is intentional: status must stay distinguishable without color.
  good: 'rounded-full bg-primary',
  muted: 'rounded-full border border-current bg-transparent text-muted-foreground/60',
  warn: 'rotate-45 rounded-[1px] bg-(--ui-yellow)',
  bad: 'rounded-[1px] bg-destructive'
}

interface StatusDotProps extends ComponentProps<'span'> {
  tone: StatusTone
}

/**
 * Compact status mark. Tone is encoded by both color and shape:
 * good = filled circle, warning = diamond, bad = square, muted = hollow circle.
 * Callers still provide nearby visible copy; this mark is supplementary.
 */
export const StatusDot = memo(function StatusDot({ className, tone, ...props }: StatusDotProps) {
  return (
    <span
      aria-hidden="true"
      className={cn('inline-block size-2 shrink-0', TONE_STYLE[tone], className)}
      data-status-tone={tone}
      {...props}
    />
  )
})
