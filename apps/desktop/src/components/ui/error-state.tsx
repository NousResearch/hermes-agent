import type { ReactNode } from 'react'

import { AlertCircle } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface ErrorStateProps {
  /** Optional actions row/stack rendered below the copy. */
  children?: ReactNode
  className?: string
  description?: ReactNode
  /** Defaults to a destructive AlertCircle. */
  icon?: ReactNode
  title: ReactNode
}

// Shared, presentation-only error layout: a destructive icon chip over a
// centered title + body, with an optional actions stack. Used by both the
// top-level React error boundary and the in-dialog update error so every
// failure state reads the same. Title/description accept nodes so callers in a
// Radix Dialog can pass DialogTitle/DialogDescription for accessibility.
export function ErrorState({ children, className, description, icon, title }: ErrorStateProps) {
  return (
    <div className={cn('grid gap-5', className)}>
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="flex size-14 items-center justify-center rounded-2xl bg-destructive/10 text-destructive">
          {icon ?? <AlertCircle className="size-7" />}
        </span>

        {typeof title === 'string' ? (
          <h2 className="text-center text-xl font-semibold tracking-tight">{title}</h2>
        ) : (
          title
        )}

        {typeof description === 'string' ? (
          <p className="max-w-prose text-center text-sm leading-5 text-muted-foreground">{description}</p>
        ) : (
          description
        )}
      </div>

      {children && <div className="grid gap-2">{children}</div>}
    </div>
  )
}
