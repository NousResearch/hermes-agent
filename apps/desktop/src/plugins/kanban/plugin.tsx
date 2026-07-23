import {
  Badge,
  Button,
  cn,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  ErrorState,
  host,
  Input,
  Loader,
  Textarea,
  useValue
} from '@hermes/plugin-sdk'
import type { HermesPlugin, PluginContext } from '@hermes/plugin-sdk'
import * as React from 'react'

interface KanbanTask {
  assignee?: string | null
  body?: string | null
  description?: string | null
  id: string | number
  latest_result?: unknown
  latest_summary?: unknown
  priority?: number | string | null
  result?: unknown
  status?: string | null
  title?: string | null
}

interface KanbanColumn {
  name: string
  tasks: KanbanTask[]
}

interface KanbanBoard {
  columns: KanbanColumn[]
  latest_event_id?: number
}

interface KanbanTaskDetails {
  latest_result?: unknown
  latest_summary?: unknown
  task: KanbanTask
}

interface NewTaskForm {
  assignee: string
  body: string
  priority: string
  title: string
}

const AUTO_REFRESH_DELAY_MS = 30_000
const boardRequests = new WeakMap<PluginContext, Map<string, Promise<KanbanBoard>>>()

const EMPTY_NEW_TASK: NewTaskForm = { assignee: '', body: '', priority: '', title: '' }

function boardRequestFor(context: PluginContext, profile: string): Promise<KanbanBoard> | null {
  return boardRequests.get(context)?.get(profile) ?? null
}

function requestBoard(context: PluginContext, profile: string): Promise<KanbanBoard> {
  const existing = boardRequestFor(context, profile)

  if (existing) {
    return existing
  }

  const requestsForContext = boardRequests.get(context) ?? new Map<string, Promise<KanbanBoard>>()

  const request = context.rest<KanbanBoard>('/board', { timeoutMs: 60_000 }).finally(() => {
    if (requestsForContext.get(profile) === request) {
      requestsForContext.delete(profile)
    }
  })

  requestsForContext.set(profile, request)
  boardRequests.set(context, requestsForContext)

  return request
}

function displayValue(value: unknown): string | null {
  if (typeof value === 'string') {
    return value.trim() || null
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const summary = value as { message?: unknown; result?: unknown; summary?: unknown }
    const nested = summary.summary ?? summary.result ?? summary.message

    if (typeof nested === 'string' && nested.trim()) {
      return nested.trim()
    }
  }

  return null
}

function DetailRow({ label, value }: { label: string; value: unknown }) {
  const text = displayValue(value)

  if (!text) {
    return null
  }

  return (
    <div className="grid min-w-0 grid-cols-[5rem_minmax(0,1fr)] gap-3">
      <dt className="text-(--ui-text-secondary)">{label}</dt>
      <dd className="min-w-0 break-words [overflow-wrap:anywhere]">{text}</dd>
    </div>
  )
}

function KanbanPage({ context }: { context: PluginContext }) {
  const profile = useValue(host.state.profile).trim() || 'default'
  const [board, setBoard] = React.useState<KanbanBoard | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [createError, setCreateError] = React.useState<string | null>(null)
  const [createForm, setCreateForm] = React.useState<NewTaskForm>(EMPTY_NEW_TASK)
  const [createOpen, setCreateOpen] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [selectedTask, setSelectedTask] = React.useState<KanbanTask | null>(null)
  const [taskDetails, setTaskDetails] = React.useState<KanbanTaskDetails | null>(null)
  const [taskDetailsError, setTaskDetailsError] = React.useState<string | null>(null)
  const [taskDetailsLoading, setTaskDetailsLoading] = React.useState(false)
  const autoRefreshTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const requestGeneration = React.useRef(0)
  const taskDetailsGeneration = React.useRef(0)

  const load = React.useCallback(
    async (showLoading = true) => {
      const generation = ++requestGeneration.current

      if (showLoading) {
        setLoading(true)
      }

      setError(null)

      try {
        // Shared boards can exceed the desktop bridge's generic 15 s timeout.
        const value = await requestBoard(context, profile)

        if (requestGeneration.current === generation) {
          setBoard(value)
        }
      } catch (reason) {
        if (requestGeneration.current === generation) {
          setError(reason instanceof Error ? reason.message : String(reason))
        }
      } finally {
        if (showLoading && requestGeneration.current === generation) {
          setLoading(false)
        }
      }
    },
    [context, profile]
  )

  const openTaskDetails = React.useCallback(
    async (task: KanbanTask) => {
      const generation = ++taskDetailsGeneration.current

      setSelectedTask(task)
      setTaskDetails(null)
      setTaskDetailsError(null)
      setTaskDetailsLoading(true)

      try {
        const value = await context.rest<KanbanTaskDetails>(`/tasks/${encodeURIComponent(String(task.id))}`, {
          timeoutMs: 60_000
        })

        if (taskDetailsGeneration.current === generation) {
          setTaskDetails(value)
        }
      } catch (reason) {
        if (taskDetailsGeneration.current === generation) {
          setTaskDetailsError(reason instanceof Error ? reason.message : String(reason))
        }
      } finally {
        if (taskDetailsGeneration.current === generation) {
          setTaskDetailsLoading(false)
        }
      }
    },
    [context]
  )

  const submitNewTask = React.useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault()

      const title = createForm.title.trim()
      const body = createForm.body.trim()
      const assignee = createForm.assignee.trim()
      const priorityText = createForm.priority.trim()

      if (!title) {
        setCreateError('Title is required.')

        return
      }

      let priority: number | undefined

      if (priorityText) {
        priority = Number(priorityText)

        if (!Number.isFinite(priority)) {
          setCreateError('Priority must be a number.')

          return
        }
      }

      const bodyPayload: { assignee?: string; body?: string; priority?: number; title: string } = { title }

      if (body) {
        bodyPayload.body = body
      }

      if (assignee) {
        bodyPayload.assignee = assignee
      }

      if (priority !== undefined) {
        bodyPayload.priority = priority
      }

      setSaving(true)
      setCreateError(null)

      try {
        await context.rest<{ task: KanbanTask; warning?: string }>('/tasks', {
          body: bodyPayload,
          method: 'POST'
        })
        setCreateOpen(false)
        setCreateForm(EMPTY_NEW_TASK)
        await load()
      } catch (reason) {
        setCreateError(reason instanceof Error ? reason.message : String(reason))
      } finally {
        setSaving(false)
      }
    },
    [context, createForm, load]
  )

  React.useEffect(() => {
    void load()

    const disposeSocket = context.socket('/events', () => {
      if (boardRequestFor(context, profile) || autoRefreshTimer.current) {
        return
      }

      autoRefreshTimer.current = setTimeout(() => {
        autoRefreshTimer.current = null
        void load(false)
      }, AUTO_REFRESH_DELAY_MS)
    })

    return () => {
      requestGeneration.current += 1
      disposeSocket()

      if (autoRefreshTimer.current) {
        clearTimeout(autoRefreshTimer.current)
        autoRefreshTimer.current = null
      }
    }
  }, [context, load, profile])

  if (loading && !board) {
    return (
      <div className="grid h-full place-items-center p-8">
        <Loader aria-label="Loading Kanban" className="size-6 text-(--ui-text-secondary)" />
      </div>
    )
  }

  if (error && !board) {
    return (
      <div className="grid h-full place-items-center p-8">
        <ErrorState description={error} title="Kanban could not be loaded">
          <Button onClick={() => void load()} size="sm" type="button">
            Retry
          </Button>
        </ErrorState>
      </div>
    )
  }

  const detailTask = taskDetails?.task ?? selectedTask
  const detailBody = detailTask?.body ?? detailTask?.description
  const detailSummary = detailTask?.latest_summary ?? taskDetails?.latest_summary
  const detailResult = detailTask?.result ?? detailTask?.latest_result ?? taskDetails?.latest_result

  return (
    <div className="kanban-route flex h-full min-h-0 min-w-0 max-w-full flex-col gap-4 overflow-hidden p-4">
      <div className="kanban-header flex min-w-0 max-w-full shrink-0 items-center gap-3">
        <div className="min-w-0 max-w-full">
          <h1 className="text-lg font-semibold">Kanban</h1>
          <p className="max-w-full truncate text-xs text-(--ui-text-secondary)">Tasks from the active Hermes profile</p>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <Button
            onClick={() => {
              setCreateError(null)
              setCreateForm(EMPTY_NEW_TASK)
              setCreateOpen(true)
            }}
            size="sm"
            type="button"
          >
            New task
          </Button>
          <Button
            disabled={loading}
            onClick={() => {
              if (autoRefreshTimer.current) {
                clearTimeout(autoRefreshTimer.current)
                autoRefreshTimer.current = null
              }

              void load()
            }}
            size="sm"
            type="button"
            variant="outline"
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </Button>
        </div>
      </div>

      {error && <div className="text-xs text-destructive">Refresh failed: {error}</div>}

      <div className="kanban-board-scroller min-h-0 min-w-0 max-w-full flex-1 overflow-x-auto overflow-y-hidden pb-2">
        <div className="kanban-board-grid grid h-full min-h-0 w-max min-w-full auto-cols-[minmax(14rem,16rem)] grid-flow-col items-stretch gap-3">
          {(board?.columns ?? []).map(column => (
            <section
              className="kanban-column flex h-full min-h-0 min-w-0 max-w-full flex-col overflow-hidden rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) p-3"
              key={column.name}
            >
              <div className="mb-3 flex min-w-0 max-w-full shrink-0 items-center justify-between gap-2">
                <h2 className="min-w-0 max-w-full truncate text-sm font-medium capitalize">{column.name}</h2>
                <Badge className="shrink-0" size="xs" variant="muted">
                  {column.tasks.length}
                </Badge>
              </div>
              <div className="kanban-task-list grid min-h-0 min-w-0 max-w-full flex-1 content-start gap-2 overflow-x-hidden overflow-y-auto pr-1">
                {column.tasks.map(task => (
                  <Button
                    aria-label={`Open task ${task.title || task.id}`}
                    className={cn(
                      'kanban-task-card flex h-auto w-full min-w-0 max-w-full flex-col items-start justify-start overflow-hidden rounded border border-(--ui-stroke-tertiary) bg-(--ui-bg-primary) p-2 text-left whitespace-normal hover:border-primary/60 hover:bg-(--chrome-action-hover) focus-visible:border-primary'
                    )}
                    key={task.id}
                    onClick={() => void openTaskDetails(task)}
                    type="button"
                    variant="ghost"
                  >
                    <span className="min-w-0 max-w-full break-words text-xs font-medium [overflow-wrap:anywhere]">
                      {task.title || task.id}
                    </span>
                    {task.assignee && (
                      <span className="mt-1 min-w-0 max-w-full break-words text-[0.68rem] text-(--ui-text-secondary) [overflow-wrap:anywhere]">
                        {task.assignee}
                      </span>
                    )}
                  </Button>
                ))}
                {column.tasks.length === 0 && (
                  <div className="min-w-0 max-w-full text-xs text-(--ui-text-secondary)">No tasks</div>
                )}
              </div>
            </section>
          ))}
        </div>
      </div>

      <Dialog
        onOpenChange={open => {
          if (!open && !saving) {
            setCreateOpen(false)
            setCreateError(null)
          }
        }}
        open={createOpen}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New task</DialogTitle>
            <DialogDescription>Create a task in the active Hermes profile.</DialogDescription>
          </DialogHeader>
          <form className="grid min-w-0 gap-3" onSubmit={event => void submitNewTask(event)}>
            <label className="grid gap-1 text-xs font-medium" htmlFor="kanban-new-task-title">
              Title
              <Input
                autoFocus
                id="kanban-new-task-title"
                onChange={event => setCreateForm(value => ({ ...value, title: event.target.value }))}
                onInvalid={() => setCreateError('Title is required.')}
                required
                value={createForm.title}
              />
            </label>
            <label className="grid gap-1 text-xs font-medium" htmlFor="kanban-new-task-body">
              Description <span className="font-normal text-(--ui-text-secondary)">(optional)</span>
              <Textarea
                id="kanban-new-task-body"
                onChange={event => setCreateForm(value => ({ ...value, body: event.target.value }))}
                value={createForm.body}
              />
            </label>
            <label className="grid gap-1 text-xs font-medium" htmlFor="kanban-new-task-assignee">
              Assignee <span className="font-normal text-(--ui-text-secondary)">(optional)</span>
              <Input
                id="kanban-new-task-assignee"
                onChange={event => setCreateForm(value => ({ ...value, assignee: event.target.value }))}
                value={createForm.assignee}
              />
            </label>
            <label className="grid gap-1 text-xs font-medium" htmlFor="kanban-new-task-priority">
              Priority <span className="font-normal text-(--ui-text-secondary)">(optional number)</span>
              <Input
                id="kanban-new-task-priority"
                inputMode="numeric"
                onChange={event => setCreateForm(value => ({ ...value, priority: event.target.value }))}
                type="number"
                value={createForm.priority}
              />
            </label>
            {createError && (
              <div className="text-xs text-destructive" role="alert">
                {createError}
              </div>
            )}
            <DialogFooter>
              <Button disabled={saving} onClick={() => setCreateOpen(false)} type="button" variant="outline">
                Cancel
              </Button>
              <Button disabled={saving} type="submit">
                {saving ? 'Creating…' : 'Create task'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        onOpenChange={open => {
          if (!open) {
            taskDetailsGeneration.current += 1
            setSelectedTask(null)
            setTaskDetails(null)
            setTaskDetailsError(null)
            setTaskDetailsLoading(false)
          }
        }}
        open={selectedTask !== null}
      >
        <DialogContent className="max-h-[calc(100vh-4rem)] max-w-xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{detailTask?.title || 'Task details'}</DialogTitle>
            <DialogDescription>Task details from the active Hermes profile.</DialogDescription>
          </DialogHeader>
          {taskDetailsLoading && (
            <div className="grid place-items-center py-8">
              <Loader aria-label="Loading task details" className="size-5 text-(--ui-text-secondary)" />
            </div>
          )}
          {taskDetailsError && !taskDetailsLoading && (
            <ErrorState description={taskDetailsError} title="Task details could not be loaded">
              <Button onClick={() => selectedTask && void openTaskDetails(selectedTask)} size="sm" type="button">
                Retry
              </Button>
            </ErrorState>
          )}
          {detailTask && !taskDetailsLoading && !taskDetailsError && (
            <div className="grid min-w-0 gap-4 text-xs">
              <div className="grid min-w-0 gap-2">
                <DetailRow label="ID" value={detailTask.id} />
                <DetailRow label="Status" value={detailTask.status} />
                <DetailRow label="Assignee" value={detailTask.assignee} />
                <DetailRow label="Priority" value={detailTask.priority} />
              </div>
              {displayValue(detailBody) && (
                <div className="min-w-0">
                  <h3 className="mb-1 font-medium">Description</h3>
                  <p className="whitespace-pre-wrap break-words text-(--ui-text-secondary) [overflow-wrap:anywhere]">
                    {displayValue(detailBody)}
                  </p>
                </div>
              )}
              {displayValue(detailSummary) && (
                <div className="min-w-0">
                  <h3 className="mb-1 font-medium">Latest summary</h3>
                  <p className="whitespace-pre-wrap break-words text-(--ui-text-secondary) [overflow-wrap:anywhere]">
                    {displayValue(detailSummary)}
                  </p>
                </div>
              )}
              {displayValue(detailResult) && (
                <div className="min-w-0">
                  <h3 className="mb-1 font-medium">Latest result</h3>
                  <p className="whitespace-pre-wrap break-words text-(--ui-text-secondary) [overflow-wrap:anywhere]">
                    {displayValue(detailResult)}
                  </p>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

const kanbanPlugin: HermesPlugin = {
  id: 'kanban',
  name: 'Kanban',
  register(context) {
    context.register({
      area: 'routes',
      data: { path: '/kanban' },
      id: 'route',
      render: () => <KanbanPage context={context} />,
      title: 'Kanban'
    })
    context.register({
      area: 'sidebar.nav',
      data: { codicon: 'package', label: 'Kanban', path: '/kanban', position: 'after:skills' },
      id: 'nav',
      order: 500
    })
  }
}

export default kanbanPlugin
