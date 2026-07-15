import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  blockKanbanTask,
  completeKanbanTask,
  createKanbanTask,
  getKanbanTasks,
  unblockKanbanTask,
  type KanbanTask
} from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { EmptyState, RightSidebarSectionHeader } from './index'

const VISIBLE_STATUSES = ['running', 'ready', 'todo', 'blocked', 'triage', 'review', 'scheduled'] as const

type VisibleStatus = (typeof VISIBLE_STATUSES)[number]

const STATUS_LABELS: Record<string, string> = {
  running: 'Doing',
  ready: 'Ready',
  todo: 'Todo',
  blocked: 'Blocked',
  triage: 'Triage',
  review: 'Review',
  scheduled: 'Scheduled',
  done: 'Done',
  archived: 'Archived'
}

const STATUS_CLASS: Record<string, string> = {
  running: 'border-emerald-400/45 bg-emerald-400/12 text-emerald-200',
  ready: 'border-sky-400/45 bg-sky-400/12 text-sky-200',
  todo: 'border-(--ui-stroke-tertiary) bg-(--ui-control-background) text-(--ui-text-secondary)',
  blocked: 'border-amber-400/45 bg-amber-400/12 text-amber-200',
  triage: 'border-fuchsia-400/45 bg-fuchsia-400/12 text-fuchsia-200',
  review: 'border-violet-400/45 bg-violet-400/12 text-violet-200',
  scheduled: 'border-(--ui-stroke-tertiary) bg-(--ui-control-background) text-(--ui-text-tertiary)'
}

function taskTitle(task: KanbanTask): string {
  return String(task?.title || '').trim() || '(untitled task)'
}

function taskBody(task: KanbanTask): string {
  return String(task?.body || '').trim()
}

function sortTasks(a: KanbanTask, b: KanbanTask): number {
  const pa = Number(a.priority || 0)
  const pb = Number(b.priority || 0)

  if (pa !== pb) {
    return pb - pa
  }

  return Number(a.created_at || 0) - Number(b.created_at || 0)
}

function groupTasks(tasks: KanbanTask[]): Map<VisibleStatus, KanbanTask[]> {
  const groups = new Map<VisibleStatus, KanbanTask[]>(VISIBLE_STATUSES.map(status => [status, []]))

  for (const task of tasks) {
    const status = String(task.status || 'todo') as VisibleStatus

    if (!groups.has(status)) {
      continue
    }

    groups.get(status)?.push(task)
  }

  for (const rows of groups.values()) {
    rows.sort(sortTasks)
  }

  return groups
}

function TaskBadge({ status }: { status: string }) {
  const label = STATUS_LABELS[status] || status

  return (
    <span
      className={cn(
        'rounded-full border px-1.5 py-0.5 text-[0.58rem] font-semibold uppercase tracking-[0.08em]',
        STATUS_CLASS[status] || STATUS_CLASS.todo
      )}
    >
      {label}
    </span>
  )
}

function TaskRow({
  busy,
  onBlock,
  onComplete,
  onUnblock,
  task
}: {
  busy: boolean
  onBlock: (task: KanbanTask) => void
  onComplete: (task: KanbanTask) => void
  onUnblock: (task: KanbanTask) => void
  task: KanbanTask
}) {
  const blocked = task.status === 'blocked' || task.status === 'scheduled'

  return (
    <div className="group/task rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background)/70 p-2 shadow-sm">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <TaskBadge status={task.status} />
            <span className="truncate font-mono text-[0.6rem] text-(--ui-text-quaternary)">{task.id}</span>
          </div>
        </div>
        <div className="flex shrink-0 gap-1 opacity-0 transition-opacity group-hover/task:opacity-100 group-focus-within/task:opacity-100">
          {blocked ? (
            <Button
              aria-label="Unblock task"
              disabled={busy}
              onClick={() => onUnblock(task)}
              size="icon-xs"
              title="Unblock"
              variant="ghost"
            >
              <Codicon name="debug-start" size="0.75rem" />
            </Button>
          ) : (
            <Button
              aria-label="Block task"
              disabled={busy}
              onClick={() => onBlock(task)}
              size="icon-xs"
              title="Block"
              variant="ghost"
            >
              <Codicon name="debug-pause" size="0.75rem" />
            </Button>
          )}
          <Button
            aria-label="Complete task"
            disabled={busy}
            onClick={() => onComplete(task)}
            size="icon-xs"
            title="Complete"
            variant="ghost"
          >
            <Codicon name="check" size="0.75rem" />
          </Button>
        </div>
      </div>
      <div className="mt-1 line-clamp-2 text-[0.72rem] font-medium leading-snug text-(--ui-text-primary)" title={taskTitle(task)}>
        {taskTitle(task)}
      </div>
      {taskBody(task) && (
        <div className="mt-1 line-clamp-2 text-[0.66rem] leading-snug text-(--ui-text-tertiary)" title={taskBody(task)}>
          {taskBody(task)}
        </div>
      )}
      {task.assignee && <div className="mt-1 text-[0.6rem] text-(--ui-text-quaternary)">@{task.assignee}</div>}
    </div>
  )
}

export function TaskPanel() {
  const [collapsed, setCollapsed] = useState(false)
  const [tasks, setTasks] = useState<KanbanTask[]>([])
  const [board, setBoard] = useState('default')
  const [loading, setLoading] = useState(true)
  const [busyTaskId, setBusyTaskId] = useState<null | string>(null)
  const [adding, setAdding] = useState(false)
  const [newTitle, setNewTitle] = useState('')

  const refresh = useCallback(async () => {
    try {
      const data = await getKanbanTasks()
      setBoard(data.board || 'default')
      setTasks(Array.isArray(data.tasks) ? data.tasks : [])
    } catch (error) {
      notifyError(error, 'Failed to load task panel')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const id = window.setInterval(() => void refresh(), 10_000)
    return () => window.clearInterval(id)
  }, [refresh])

  const visibleTasks = useMemo(
    () => tasks.filter(task => VISIBLE_STATUSES.includes(task.status as VisibleStatus)),
    [tasks]
  )
  const groups = useMemo(() => groupTasks(visibleTasks), [visibleTasks])
  const total = visibleTasks.length

  async function handleAdd(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const title = newTitle.trim()

    if (!title) {
      return
    }

    setAdding(true)
    try {
      const result = await createKanbanTask({ status: 'ready', title })
      setTasks(rows => [result.task, ...rows.filter(row => row.id !== result.task.id)])
      setNewTitle('')
      notify({ kind: 'success', title: 'Task added', message: title.slice(0, 80) })
    } catch (error) {
      notifyError(error, 'Failed to add task')
    } finally {
      setAdding(false)
    }
  }

  async function mutateTask(
    task: KanbanTask,
    label: string,
    fn: (task: KanbanTask) => Promise<{ task: KanbanTask }>
  ) {
    setBusyTaskId(task.id)
    try {
      const result = await fn(task)
      setTasks(rows => rows.map(row => (row.id === task.id ? result.task : row)))
      notify({ kind: 'success', title: label, message: taskTitle(task).slice(0, 80) })
    } catch (error) {
      notifyError(error, `Failed to ${label.toLowerCase()}`)
    } finally {
      setBusyTaskId(null)
    }
  }

  const handleComplete = (task: KanbanTask) =>
    mutateTask(task, 'Task completed', t => completeKanbanTask(t.id, 'Completed from desktop task panel'))
  const handleBlock = (task: KanbanTask) =>
    mutateTask(task, 'Task blocked', t => blockKanbanTask(t.id, 'Blocked from desktop task panel'))
  const handleUnblock = (task: KanbanTask) => mutateTask(task, 'Task unblocked', t => unblockKanbanTask(t.id))

  return (
    <section
      className={cn(
        'flex shrink-0 flex-col border-b border-(--ui-stroke-secondary)',
        collapsed ? 'min-h-0' : 'min-h-[13rem] flex-1'
      )}
    >
      <RightSidebarSectionHeader className="gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-[0.66rem] font-semibold uppercase tracking-[0.11em] text-(--ui-text-secondary)">Tasks</span>
            <span className="rounded-full bg-(--ui-control-background) px-1.5 py-0.5 text-[0.58rem] text-(--ui-text-quaternary)">
              {total}
            </span>
          </div>
          {!collapsed && <div className="truncate text-[0.56rem] text-(--ui-text-quaternary)">board: {board}</div>}
        </div>
        <Button
          aria-label={collapsed ? 'Expand tasks' : 'Collapse tasks'}
          onClick={() => setCollapsed(v => !v)}
          size="icon-xs"
          title={collapsed ? 'Expand tasks' : 'Collapse tasks'}
          variant="ghost"
        >
          <Codicon name={collapsed ? 'chevron-down' : 'chevron-up'} size="0.8125rem" />
        </Button>
        <Button
          aria-label="Refresh tasks"
          disabled={loading}
          onClick={() => void refresh()}
          size="icon-xs"
          title="Refresh tasks"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.8125rem" spinning={loading} />
        </Button>
      </RightSidebarSectionHeader>
      {!collapsed && (
        <form className="flex shrink-0 gap-1 px-2 pb-2" onSubmit={handleAdd}>
          <input
            className="min-w-0 flex-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-input-background) px-2 py-1 text-[0.68rem] text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-quaternary) focus:border-(--ui-accent)"
            disabled={adding}
            onChange={event => setNewTitle(event.currentTarget.value)}
            placeholder="Add durable task…"
            value={newTitle}
          />
          <Button disabled={adding || !newTitle.trim()} size="xs" type="submit" variant="outline">
            Add
          </Button>
        </form>
      )}
      {!collapsed && (
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {loading && tasks.length === 0 ? (
            <EmptyState body="Loading durable Kanban tasks…" title="Task panel" />
          ) : total === 0 ? (
            <EmptyState body="Add a task here or ask Hermes to create Kanban cards." title="No active tasks" />
          ) : (
            <div className="space-y-2">
              {VISIBLE_STATUSES.map(status => {
                const rows = groups.get(status) || []

                if (!rows.length) {
                  return null
                }

                return (
                  <div className="space-y-1" key={status}>
                    <div className="sticky top-0 z-10 bg-(--ui-sidebar-surface-background)/95 py-1 text-[0.58rem] font-semibold uppercase tracking-[0.12em] text-(--ui-text-quaternary) backdrop-blur">
                      {STATUS_LABELS[status] || status} · {rows.length}
                    </div>
                    {rows.map(task => (
                      <TaskRow
                        busy={busyTaskId === task.id}
                        key={task.id}
                        onBlock={handleBlock}
                        onComplete={handleComplete}
                        onUnblock={handleUnblock}
                        task={task}
                      />
                    ))}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
