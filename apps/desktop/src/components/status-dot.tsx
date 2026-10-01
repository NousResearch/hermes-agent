import type { ComponentProps } from 'react'
import { memo } from 'react'

import { cn } from '@/lib/utils'

export type StatusTone = 'good' | 'muted' | 'warn' | 'bad'
export type StatusShape = 'circle' | 'diamond' | 'ring' | 'square'

// Shape is the channel that survives color-vision deficiency, so no two tones
// share one. The diamond and square scale down inside the same 6px box so their
// optical weight matches the circle without moving anything around them.
export const STATUS_SHAPE_CLASS: Record<StatusShape, string> = {
  circle: 'rounded-full',
  diamond: 'rotate-45 scale-[0.9] rounded-[0.5px]',
  ring: 'rounded-full border border-current',
  square: 'scale-[0.85] rounded-[1px]'
}

export const STATUS_TONE_SHAPE: Record<StatusTone, StatusShape> = {
  bad: 'square',
  good: 'circle',
  muted: 'ring',
  warn: 'diamond'
}

const TONE_COLOR: Record<StatusTone, string> = {
  bad: 'bg-(--ui-status-danger)',
  good: 'bg-(--ui-status-success)',
  muted: 'text-muted-foreground/75',
  warn: 'bg-(--ui-status-warning)'
}

interface StatusDotProps extends ComponentProps<'span'> {
  tone: StatusTone
}

/** Compact status mark, supplementary to nearby visible copy. */
export const StatusDot = memo(function StatusDot({ className, tone, ...props }: StatusDotProps) {
  const shape = STATUS_TONE_SHAPE[tone]

  return (
    <span
      aria-hidden="true"
      className={cn('inline-block size-1.5 shrink-0', STATUS_SHAPE_CLASS[shape], TONE_COLOR[tone], className)}
      data-status-shape={shape}
      data-status-tone={tone}
      {...props}
    />
  )
})
