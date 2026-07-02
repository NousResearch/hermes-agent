import { type DraggableAttributes, type DraggableSyntheticListeners, useDraggable } from '@dnd-kit/core'
import type { Ref } from 'react'

import { Badge } from '@/components/ui/badge'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import type { KanbanTask } from '@/types/hermes'

import { formatAge } from './helpers'

interface KanbanCardContentProps {
  task: KanbanTask
  now: number
  dragging?: boolean
  onOpen?: (taskId: string) => void
  dragRef?: Ref<HTMLButtonElement>
  dragAttributes?: DraggableAttributes
  dragListeners?: DraggableSyntheticListeners
}

export function KanbanCardContent({
  task,
  now,
  dragging = false,
  onOpen,
  dragRef,
  dragAttributes,
  dragListeners
}: KanbanCardContentProps) {
  const age = formatAge(task.created_at, now)
  const hasWarnings = (task.warnings?.count ?? 0) > 0

  return (
    <button
      className={cn(
        'flex w-full touch-none flex-col gap-1.5 rounded-[6px] border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) px-2.5 py-2 text-left transition-colors',
        'hover:border-(--ui-stroke-secondary) focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-(--ui-accent)',
        dragging && 'shadow-nous border-(--stroke-nous)'
      )}
      onClick={onOpen ? () => onOpen(task.id) : undefined}
      ref={dragRef}
      type="button"
      {...dragAttributes}
      {...dragListeners}
    >
      <span className="line-clamp-3 text-xs leading-snug text-(--ui-text-primary)">{task.title}</span>

      {task.latest_summary ? (
        <span className="line-clamp-2 text-[0.68rem] leading-snug text-(--ui-text-tertiary)">
          {task.latest_summary}
        </span>
      ) : null}

      <div className="flex flex-wrap items-center gap-1">
        {task.assignee ? (
          <Badge variant="muted">
            <Codicon name="account" /> {task.assignee}
          </Badge>
        ) : null}

        {task.progress && task.progress.total > 0 ? (
          <Badge variant="outline">
            {task.progress.done}/{task.progress.total}
          </Badge>
        ) : null}

        {task.priority > 0 ? <Badge variant="outline">P{task.priority}</Badge> : null}

        {task.status === 'running' && task.worker_pid ? (
          <Badge variant="default">
            <Codicon name="pulse" /> live
          </Badge>
        ) : null}

        {hasWarnings ? (
          <Badge variant="warn">
            <Codicon name="warning" /> {task.warnings?.count}
          </Badge>
        ) : null}

        {age ? <span className="ml-auto text-[0.62rem] text-(--ui-text-tertiary)">{age}</span> : null}
      </div>
    </button>
  )
}

interface KanbanCardProps {
  task: KanbanTask
  now: number
  onOpen: (taskId: string) => void
}

export function KanbanCard({ task, now, onOpen }: KanbanCardProps) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: task.id,
    data: { status: task.status }
  })

  return (
    <div className={cn(isDragging && 'opacity-40')}>
      <KanbanCardContent
        dragAttributes={attributes}
        dragListeners={listeners}
        dragRef={setNodeRef}
        now={now}
        onOpen={onOpen}
        task={task}
      />
    </div>
  )
}
