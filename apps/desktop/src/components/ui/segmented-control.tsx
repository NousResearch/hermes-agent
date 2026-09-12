import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface SegmentedControlOption<T extends string> {
  id: T
  label: string
  icon?: IconComponent
}

interface SegmentedControlProps<T extends string> {
  options: readonly SegmentedControlOption<T>[]
  value: T
  onChange: (id: T) => void
  className?: string
  /** Dims the whole track and blocks selection (e.g. gated behind a prerequisite). */
  disabled?: boolean
  /** What this track is choosing. Without it a screen reader reads a bare run of
   *  toggle buttons — "Light 버튼, Dark 버튼" — with nothing saying they are the
   *  colour-mode setting (SenseReader, 2026-09-06). A settings `ListRow` supplies
   *  `aria-labelledby` automatically by pointing at the row title, so most call
   *  sites need no change; pass `aria-label` for a track outside a row. */
  'aria-label'?: string
  'aria-labelledby'?: string
}

/**
 * Grouped one-row toggle used for small mutually-exclusive choices
 * (color mode, tool-call display, usage period, etc.). Flat by design —
 * no per-option borders, just a tinted track with a raised active pill.
 */
export function SegmentedControl<T extends string>({
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledBy,
  className,
  disabled = false,
  onChange,
  options,
  value
}: SegmentedControlProps<T>) {
  return (
    <div
      // `group`, not `radiogroup`: these are Tab-reachable toggle buttons, and
      // a radiogroup promises arrow-key navigation this does not implement —
      // claiming the wrong role reads worse than claiming none.
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      className={cn(
        'inline-grid w-fit auto-cols-fr grid-flow-col gap-0.5 rounded-[5px] bg-(--ui-bg-tertiary) p-0.5',
        disabled && 'opacity-50',
        className
      )}
      role="group"
    >
      {options.map(({ id, label, icon: Icon }) => {
        const active = value === id

        return (
          <button
            aria-pressed={active}
            className={cn(
              'flex items-center justify-center gap-1 rounded-[3px] px-2.5 py-0.5 text-[0.6875rem] font-medium transition-colors disabled:cursor-default',
              active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
            )}
            disabled={disabled}
            key={id}
            onClick={() => onChange(id)}
            type="button"
          >
            {Icon && <Icon className="size-3" />}
            {label}
          </button>
        )
      })}
    </div>
  )
}
