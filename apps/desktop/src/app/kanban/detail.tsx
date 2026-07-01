import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { KanbanTask } from '@/types/hermes'

import { KANBAN_COLUMN_LABELS } from './constants'
import { formatAge } from './helpers'

interface KanbanDetailProps {
  task: KanbanTask
  now: number
  onClose: () => void
}

export function KanbanDetail({ task, now, onClose }: KanbanDetailProps) {
  const age = formatAge(task.created_at, now)

  return (
    <aside className="shadow-nous flex h-full w-80 shrink-0 flex-col border-l border-(--stroke-nous) bg-(--ui-chat-surface-background)">
      <header className="flex items-start gap-2 px-4 pb-3 pt-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="outline">{KANBAN_COLUMN_LABELS[task.status]}</Badge>
            {age ? <span className="text-[0.68rem] text-(--ui-text-tertiary)">{age} old</span> : null}
          </div>
          <h2 className="mt-1.5 text-sm leading-snug text-(--ui-text-primary)">{task.title}</h2>
        </div>

        <Button aria-label="Close" onClick={onClose} size="icon-xs" title="Close" type="button" variant="ghost">
          <Codicon name="close" size="0.875rem" />
        </Button>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4">
        <dl className="flex flex-col gap-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <dt className="text-(--ui-text-tertiary)">Assignee</dt>
            <dd className="text-(--ui-text-secondary)">{task.assignee ?? 'unassigned'}</dd>
          </div>

          <div className="flex items-center justify-between gap-2">
            <dt className="text-(--ui-text-tertiary)">Priority</dt>
            <dd className="text-(--ui-text-secondary)">{task.priority}</dd>
          </div>

          {task.progress && task.progress.total > 0 ? (
            <div className="flex items-center justify-between gap-2">
              <dt className="text-(--ui-text-tertiary)">Subtasks</dt>
              <dd className="text-(--ui-text-secondary)">
                {task.progress.done}/{task.progress.total}
              </dd>
            </div>
          ) : null}

          <div className="flex items-center justify-between gap-2">
            <dt className="text-(--ui-text-tertiary)">ID</dt>
            <dd className="font-mono text-[0.68rem] text-(--ui-text-tertiary)">{task.id}</dd>
          </div>
        </dl>

        {task.latest_summary ? (
          <div className="flex flex-col gap-1">
            <span className="text-[0.68rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
              Latest handoff
            </span>
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-(--ui-text-secondary)">
              {task.latest_summary}
            </p>
          </div>
        ) : null}

        {task.body ? (
          <div className="flex flex-col gap-1">
            <span className="text-[0.68rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
              Description
            </span>
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-(--ui-text-secondary)">{task.body}</p>
          </div>
        ) : null}
      </div>
    </aside>
  )
}
