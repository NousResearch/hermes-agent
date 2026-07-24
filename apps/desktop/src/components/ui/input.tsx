import * as React from 'react'

import { cn } from '@/lib/utils'

import { type ControlVariantProps, controlVariants } from './control'

// `prefix`/`suffix` are DOM string attributes on the native element; we shadow
// them with ReactNode adornments, so they're omitted from the base props.
type InputProps = Omit<React.ComponentProps<'input'>, 'size' | 'prefix' | 'suffix'> &
  ControlVariantProps & {
    /** Leading adornment rendered inside the field (e.g. a `$` for money). */
    prefix?: React.ReactNode
    /** Trailing adornment rendered inside the field (e.g. a unit label). */
    suffix?: React.ReactNode
    /** Applied to the wrapper when an adornment promotes the field to a group. */
    containerClassName?: string
  }

function Input({ className, containerClassName, prefix, suffix, size, type, ...props }: InputProps) {
  const grouped = prefix != null || suffix != null

  const field = (
    <input
      // Off by default for every consumer — these are code/config/search fields,
      // not prose. Callers can re-enable per-instance by passing the prop.
      autoCapitalize="off"
      autoComplete="off"
      autoCorrect="off"
      className={cn(
        // When adorned, the wrapper owns the chrome (border/background/focus
        // glow) and the input goes transparent so the whole thing reads as one
        // box; otherwise the input carries the chrome itself.
        grouped
          ? 'min-w-0 flex-1 border-0 bg-transparent p-0 text-xs leading-4 text-foreground outline-none placeholder:text-muted-foreground'
          : controlVariants({ size }),
        'selection:bg-primary selection:text-primary-foreground file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-xs file:font-medium file:text-foreground',
        className
      )}
      data-slot="input"
      spellCheck={false}
      type={type}
      {...props}
    />
  )

  if (!grouped) {
    return field
  }

  return (
    <div
      className={cn(
        // Same control chrome/sizing as a bare input; `.desktop-input-chrome`
        // lights on `:focus-within` (styles.css) since the div never focuses.
        controlVariants({ size }),
        'inline-flex items-center gap-1',
        props.disabled && 'cursor-not-allowed opacity-50',
        containerClassName
      )}
      data-slot="input-group"
    >
      {prefix != null && (
        <span className="pointer-events-none shrink-0 select-none text-muted-foreground">{prefix}</span>
      )}
      {field}
      {suffix != null && (
        <span className="pointer-events-none shrink-0 select-none text-muted-foreground">{suffix}</span>
      )}
    </div>
  )
}

export { Input }
