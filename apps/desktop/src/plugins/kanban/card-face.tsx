import { cn } from '@hermes/plugin-sdk'
import type { ComponentProps } from 'react'

import { columnMeta, type KanbanTask } from './types'

/** Shared card presentation. Callers own interactions and the source-correct
 *  footer; rendering a card never fetches ambient orchestration settings. */
export function CardFace({ children, className, style, task, ...props }: ComponentProps<'div'> & { task: KanbanTask }) {
  const summary = task.latest_summary || task.body

  return (
    <div
      {...props}
      className={cn(
        'group relative flex flex-col gap-2 rounded-md border border-(--ui-stroke-tertiary) border-l-2 bg-(--ui-bg-elevated) p-2.5 transition-colors hover:bg-primary/[0.06]',
        className
      )}
      style={{ borderLeftColor: columnMeta(task.status).tone, ...style }}
    >
      <span className="line-clamp-2 text-[0.8125rem] font-medium leading-snug text-foreground">
        {task.title || task.id}
      </span>
      {summary && (
        <span className="line-clamp-2 text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{summary}</span>
      )}
      {children}
    </div>
  )
}
