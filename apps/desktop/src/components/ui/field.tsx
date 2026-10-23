import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

// Shared form-field primitive for dialog forms: a label stacked above its
// control, with an optional inline "(optional)" tag. Pair with FieldHint for
// help text below the control. This is the single field language for every form
// dialog (cron, webhooks, profiles, …) — don't hand-roll label+control stacks
// or reach for the settings-surface ListRow inside a dialog. Stack Fields in a
// `grid gap-4` form; pair two across with `grid items-start gap-4 sm:grid-cols-2`.
export function Field({
  children,
  htmlFor,
  label,
  optional,
  optionalLabel
}: {
  children: ReactNode
  htmlFor?: string
  label: ReactNode
  optional?: boolean
  optionalLabel?: string
}) {
  return (
    <div className="grid gap-1.5">
      <label
        className="flex items-baseline gap-2 text-xs font-medium text-foreground"
        htmlFor={htmlFor}
        id={htmlFor ? `${htmlFor}-label` : undefined}
      >
        {label}
        {optional && optionalLabel && (
          <span className="text-[0.65rem] font-normal text-muted-foreground">{optionalLabel}</span>
        )}
      </label>
      {children}
    </div>
  )
}

export function FieldHint({ children, error }: { children: ReactNode; error?: boolean }) {
  return (
    <p className={cn('text-[0.66rem] leading-4', error ? 'text-destructive' : 'text-muted-foreground')}>{children}</p>
  )
}
