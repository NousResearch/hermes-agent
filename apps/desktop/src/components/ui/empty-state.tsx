import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

// Canonical empty state (title + one line of help + an optional action). The default for "no
// results / nothing here yet" bodies everywhere. Classic centres it in the page body; the redesign
// draws it as the empty card (Figma 877:3596 "No chats yet / Chats with this agent show up here. /
// [Start a chat]") and, inside a list card, as that card's flush first row (portal-pages.css).
export function EmptyState({
  title,
  description,
  action,
  className
}: {
  title?: ReactNode
  description?: ReactNode
  /** One labelled button that starts the missing thing ("New schedule"), never a bare glyph. */
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('grid min-h-48 place-items-center text-center', className)} data-slot="empty-state">
      <div>
        {title && (
          <div className="text-sm font-medium" data-slot="empty-state-title">
            {title}
          </div>
        )}
        {description && (
          <div className="mt-1 text-xs text-muted-foreground" data-slot="empty-state-description">
            {description}
          </div>
        )}
        {action && (
          <div className="mt-3" data-slot="empty-state-action">
            {action}
          </div>
        )}
      </div>
    </div>
  )
}
