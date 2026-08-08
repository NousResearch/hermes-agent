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
}

/**
 * Grouped one-row toggle used for small mutually-exclusive choices
 * (color mode, tool-call display, usage period, etc.). A quiet shared track and
 * raised active segment keep state readable without boxing every option.
 */
export function SegmentedControl<T extends string>({
  className,
  disabled = false,
  onChange,
  options,
  value
}: SegmentedControlProps<T>) {
  return (
    <div
      className={cn(
        'inline-grid w-fit auto-cols-fr grid-flow-col gap-0.5 rounded-[var(--radius-sm)] border border-(--ui-stroke-tertiary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_82%,transparent)] p-0.5 shadow-xs transition-opacity duration-200',
        disabled && 'opacity-50',
        className
      )}
    >
      {options.map(({ id, label, icon: Icon }) => {
        const active = value === id

        return (
          <button
            aria-pressed={active}
            className={cn(
              'flex min-h-7 items-center justify-center gap-1 rounded-[var(--radius-sm)] px-3 py-1 text-xs font-medium transition-[background-color,box-shadow,color,transform] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] active:scale-[0.98] disabled:cursor-default',
              active
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:bg-(--chrome-action-hover) hover:text-foreground'
            )}
            disabled={disabled}
            key={id}
            onClick={() => onChange(id)}
            type="button"
          >
            {Icon && <Icon className="size-3.5" />}
            {label}
          </button>
        )
      })}
    </div>
  )
}
