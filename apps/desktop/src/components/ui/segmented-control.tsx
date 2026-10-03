import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { Tip } from './tooltip'

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
  /** Render each option as its icon alone, named by tooltip + aria-label.
   *  For fixed-size tracks (narrow panes) where text labels would wrap and
   *  collide with neighbors. Options without an icon fall back to text. */
  iconOnly?: boolean
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
        // Icon-only mode keeps the track a fixed size at every width: the
        // name moves to the tooltip and the accessible name. An option
        // without an icon has nowhere to put its name, so it keeps the text.
        const unnamed = iconOnly && !!Icon

        return (
          <Tip key={id} label={unnamed ? label : ''}>
            <button
              aria-label={unnamed ? label : undefined}
              aria-pressed={active}
              className={cn(
                'flex items-center justify-center gap-1 rounded-[3px] px-2.5 py-0.5 text-[0.6875rem] font-medium transition-colors disabled:cursor-default',
                active ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
              )}
              disabled={disabled}
              onClick={() => onChange(id)}
              type="button"
            >
              {Icon && <Icon className="size-3" />}
              {!unnamed && label}
            </button>
          </Tip>
        )
      })}
    </div>
  )
}
