/**
 * A bounded number, nudged or typed.
 *
 * ONE control to the eye: the shell carries the field chrome, the − and + are
 * quiet zones inside it (full height, real hit area), the number sits between
 * them. A suffix lives inside the field too — "30 min" is one value, not a
 * number plus a caption — and the browser's own spinners are stripped, since
 * they're too small to hit and they'd be a second pair of arrows.
 */

import { type ControlVariantProps, controlVariants } from '@/components/ui/control'
import { cn } from '@/lib/utils'

export function Stepper({
  chrome,
  className,
  max = 999,
  min = 0,
  onChange,
  step = 1,
  suffix,
  unboundedAtMin,
  value
}: {
  className?: string
  max?: number
  min?: number
  onChange: (value: number) => void
  step?: number
  /** Rides inside the field, after the number — a unit, not a caption. */
  suffix?: string
  /** Render `min` as "∞", for a budget whose floor means "no limit". */
  unboundedAtMin?: boolean
  value: number
} & Pick<ControlVariantProps, 'chrome'>) {
  const clamp = (n: number) => Math.max(min, Math.min(max, n))
  const unbounded = unboundedAtMin && value <= min

  const nudge = cn(
    'shrink-0 px-1.5 text-(--ui-text-tertiary) transition-colors',
    'hover:text-foreground disabled:pointer-events-none disabled:opacity-40'
  )

  return (
    <div className={cn(controlVariants({ chrome, size: 'sm' }), 'flex items-center gap-1 py-0', className)}>
      <button
        aria-label="Decrease"
        className={nudge}
        disabled={value <= min}
        onClick={() => onChange(clamp(value - step))}
        type="button"
      >
        −
      </button>
      <input
        className="min-w-8 flex-1 border-0 bg-transparent py-1 text-center text-xs leading-4 tabular-nums outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
        max={max}
        min={min}
        onChange={event => onChange(clamp(Number(event.target.value)))}
        readOnly={unbounded}
        title={unbounded ? 'No limit' : undefined}
        type={unbounded ? 'text' : 'number'}
        value={unbounded ? '∞' : value}
      />
      {suffix && !unbounded && <span className="shrink-0 text-(--ui-text-tertiary)">{suffix}</span>}
      <button
        aria-label="Increase"
        className={nudge}
        disabled={value >= max}
        onClick={() => onChange(clamp(value + step))}
        type="button"
      >
        +
      </button>
    </div>
  )
}
