import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

// The canonical "nothing here" block: centred, quiet, and optionally offering
// the one action that would end the emptiness.
//
// Everything is optional except the title, so the same component covers the
// bare "no results" line in a settings list and the full icon + copy + button
// a page shows before its first document exists. There is no second treatment
// for the richer case — that split is what produced two of these.
export function EmptyState({
  action,
  className,
  description,
  icon,
  title
}: {
  /** The one thing to do about it. A page's first-run state should have one. */
  action?: ReactNode
  className?: string
  description?: ReactNode
  /** Codicon glyph name, e.g. 'type-hierarchy-sub'. */
  icon?: string
  title: ReactNode
}) {
  return (
    <div className={cn('grid min-h-48 place-items-center px-6 text-center', className)}>
      <div className="flex flex-col items-center gap-2">
        {icon && <Codicon className="text-muted-foreground/50" name={icon} size="1.25rem" />}
        {title && <div className="text-sm font-medium text-foreground/90">{title}</div>}
        {description && <p className="max-w-sm text-xs leading-relaxed text-muted-foreground/70">{description}</p>}
        {action && <div className="mt-2">{action}</div>}
      </div>
    </div>
  )
}
