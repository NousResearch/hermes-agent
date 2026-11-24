import * as React from 'react'

import { cn } from '@/lib/utils'

const TRACK_HEIGHT = {
  sm: 'h-1',
  default: 'h-1.5',
  lg: 'h-2'
} as const

export interface ProgressProps extends Omit<React.ComponentProps<'div'>, 'children'> {
  /** Completion as a 0–1 fraction; clamped. Ignored when `indeterminate`. */
  value?: number
  /** No known endpoint — animate instead of showing a fixed width. */
  indeterminate?: boolean
  /**
   * Use the continuous sliding animation for the indeterminate state (the pet
   * hatch look) instead of the default pulse. No effect when determinate.
   */
  animated?: boolean
  /** Swap the fill to the destructive color for error / over-limit states. */
  destructive?: boolean
  size?: keyof typeof TRACK_HEIGHT
  /** Override the fill color/shape (e.g. billing's tone mapping, pet's accent). */
  fillClassName?: string
  fillStyle?: React.CSSProperties
  /** Extra content inside the track, behind the fill (e.g. a threshold marker). */
  children?: React.ReactNode
}

/**
 * The app's one progress/meter bar: a rounded track with an animated fill. The
 * track owns `role="progressbar"` and its aria values; the fill is width-driven
 * (determinate) or animated (indeterminate). Consumers needing a bespoke look —
 * billing's hatched empty state, tone-mapped or accent fills — override
 * `className` (track) and `fillClassName` while inheriting the structure,
 * sizing, and accessibility.
 */
export function Progress({
  value = 0,
  indeterminate = false,
  animated = false,
  destructive = false,
  size = 'default',
  className,
  fillClassName,
  fillStyle,
  children,
  ...props
}: ProgressProps) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  const fillColor = destructive ? 'bg-destructive' : 'bg-primary'

  return (
    <div
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={indeterminate ? undefined : pct}
      className={cn('relative w-full overflow-hidden rounded-full bg-muted', TRACK_HEIGHT[size], className)}
      data-slot="progress"
      role="progressbar"
      {...props}
    >
      {children}
      {indeterminate && animated ? (
        // Sliding block — width/animation come from `.progress-slide` (styles.css),
        // which also honors prefers-reduced-motion.
        <div className={cn('progress-slide absolute inset-y-0 rounded-full', fillColor, fillClassName)} />
      ) : (
        <div
          className={cn(
            'relative h-full rounded-full transition-[width] duration-300 ease-out',
            fillColor,
            indeterminate && 'w-1/3 animate-pulse',
            fillClassName
          )}
          style={indeterminate ? fillStyle : { width: `${pct}%`, ...fillStyle }}
        />
      )}
    </div>
  )
}
