import { useDroppable } from '@dnd-kit/core'

import { cn } from '@/lib/utils'
import type { KanbanStatus, KanbanTask } from '@/types/hermes'

import { KanbanCard } from './card'
import { KANBAN_COLUMN_LABELS } from './constants'

interface KanbanColumnProps {
  status: KanbanStatus
  tasks: KanbanTask[]
  now: number
  onOpenTask: (taskId: string) => void
}

export function KanbanColumn({ status, tasks, now, onOpenTask }: KanbanColumnProps) {
  const { setNodeRef, isOver } = useDroppable({ id: status })

  return (
    <section className="flex h-full w-64 shrink-0 flex-col">
      <header className="flex items-center gap-2 px-1 pb-2">
        <span className="text-[0.7rem] font-medium uppercase tracking-wide text-(--ui-text-secondary)">
          {KANBAN_COLUMN_LABELS[status]}
        </span>
        <span className="text-[0.7rem] text-(--ui-text-tertiary)">{tasks.length}</span>
      </header>

      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto rounded-[8px] p-1 transition-colors',
          isOver ? 'bg-(--chrome-action-hover)' : 'bg-transparent'
        )}
        ref={setNodeRef}
      >
        {tasks.map(task => (
          <KanbanCard key={task.id} now={now} onOpen={onOpenTask} task={task} />
        ))}

        {tasks.length === 0 ? (
          <div className="flex flex-1 items-center justify-center py-6 text-[0.68rem] text-(--ui-text-tertiary)">
            —
          </div>
        ) : null}
      </div>
    </section>
  )
}
