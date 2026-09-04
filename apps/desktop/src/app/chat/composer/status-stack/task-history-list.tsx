import { Fragment } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { TodoHistorySnapshot } from '@/lib/todos'
import type { ComposerStatusItem } from '@/store/composer-status'

import { StatusItemRow } from './status-row'

interface TaskHistoryListProps {
  snapshots: readonly TodoHistorySnapshot[]
}

const historyItem = (snapshotId: string, todo: TodoHistorySnapshot['todos'][number]): ComposerStatusItem => ({
  id: `${snapshotId}:${todo.id}`,
  state: todo.status === 'in_progress' ? 'running' : 'done',
  title: todo.content,
  todoStatus: todo.status,
  type: 'todo'
})

/** Compact transcript-derived plans. The store owns ordering, retention, and
 * deduplication; this component only renders the selected session's slice. */
export function TaskHistoryList({ snapshots }: TaskHistoryListProps) {
  const { t } = useI18n()

  return (
    <div className="space-y-1 pb-0.5">
      {snapshots.map(snapshot => (
        <Fragment key={snapshot.id}>
          <div className="flex items-center gap-1.5 px-1.5 pt-1 text-[0.62rem] font-medium text-muted-foreground/70">
            <Codicon aria-hidden name={snapshot.state === 'completed' ? 'pass' : 'history'} size="0.72rem" />
            <span>
              {snapshot.state === 'completed'
                ? t.statusStack.taskHistoryCompleted
                : t.statusStack.taskHistoryUnfinished}
            </span>
          </div>
          {snapshot.todos.map(todo => (
            <StatusItemRow item={historyItem(snapshot.id, todo)} key={`${snapshot.id}:${todo.id}`} />
          ))}
        </Fragment>
      ))}
    </div>
  )
}
