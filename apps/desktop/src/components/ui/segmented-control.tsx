import { Tip } from '@/components/ui/tooltip'
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
  /** Icon buttons retain localized labels for tooltips and screen readers. */
  iconOnly?: boolean
  /** Dims the whole track and blocks selection (e.g. gated behind a prerequisite). */
  disabled?: boolean
}

/**
 * Grouped one-row toggle used for small mutually-exclusive choices
 * (color mode, tool-call display, usage period, etc.). Flat by design —
 * no per-option borders, just a tinted track with a raised active pill.
 */
export function SegmentedControl<T extends string>({
  className,
  disabled = false,
  iconOnly = false,
  onChange,
  options,
  value
}: SegmentedControlProps<T>) {
  return (
    <div
      className={cn(
        'inline-grid w-fit auto-cols-fr grid-flow-col gap-0.5 rounded-[5px] bg-(--ui-bg-tertiary) p-0.5',
        disabled && 'opacity-50',
        className
      )}
    >
      {options.map(({ id, label, icon: Icon }) => {
        const active = value === id

        const button = (
          <button
            aria-label={iconOnly ? label : undefined}
            aria-pressed={active}
            className={cn(
              'flex items-center justify-center gap-1 rounded-[3px] px-2.5 py-0.5 text-[0.6875rem] font-medium transition-colors disabled:cursor-default',
              active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
              iconOnly && Icon && 'size-6 p-0 text-sm'
            )}
            disabled={disabled}
            key={id}
            onClick={() => onChange(id)}
            type="button"
          >
            {Icon && <Icon className="size-3" />}
            {(!iconOnly || !Icon) && label}
          </button>
        )

        return iconOnly && Icon ? <Tip key={id} label={label}>{button}</Tip> : button
      })}
    </div>
  )
}
