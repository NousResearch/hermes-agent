/**
 * Native Kanban board — talks to `/api/plugins/kanban` only.
 * No web dashboard SPA / webview / serve-UI coupling.
 */
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type * as React from 'react'
import { useCallback, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  columnMap,
  createKanbanTask,
  fetchKanbanBoard,
  fetchKanbanTask,
  KANBAN_COLUMNS,
  statusLabel,
  type KanbanTask,
  updateKanbanTask
} from './api'

interface KanbanViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

const COLUMN_TONE: Record<string, string> = {
  triage: 'text-amber-300/90',
  todo: 'text-(--ui-text-secondary)',
  scheduled: 'text-sky-300/90',
  ready: 'text-emerald-300/90',
  running: 'text-violet-300/90',
  blocked: 'text-red-300/90',
  review: 'text-orange-300/90',
  done: 'text-(--ui-text-tertiary)'
}

export function KanbanView({ className, setStatusbarItemGroup: _setStatusbar, ...rest }: KanbanViewProps) {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [assignee, setAssignee] = useState('')
  const [creating, setCreating] = useState(false)
  const [busyId, setBusyId] = useState<string | null>(null)

  const boardQuery = useQuery({
    queryKey: ['kanban', 'board'],
    queryFn: () => fetchKanbanBoard(),
    refetchInterval: 15_000
  })

  const detailQuery = useQuery({
    queryKey: ['kanban', 'task', selectedId],
    queryFn: () => fetchKanbanTask(selectedId!),
    enabled: Boolean(selectedId)
  })

  const refresh = useCallback(() => {
    void qc.invalidateQueries({ queryKey: ['kanban'] })
  }, [qc])

  useRefreshHotkey(refresh)

  const columns = useMemo(() => columnMap(boardQuery.data), [boardQuery.data])

  const onCreate = async () => {
    const t = title.trim()
    if (!t || creating) return
    setCreating(true)
    try {
      const res = await createKanbanTask({
        title: t,
        body: body.trim() || undefined,
        assignee: assignee.trim() || undefined
      })
      setTitle('')
      setBody('')
      setAssignee('')
      if (res.warning) {
        notify({ message: res.warning, kind: 'warning' })
      } else {
        notify({ message: 'Task created', kind: 'success' })
      }
      if (res.task?.id) setSelectedId(res.task.id)
      refresh()
    } catch (err) {
      notifyError(err, 'Failed to create task')
    } finally {
      setCreating(false)
    }
  }

  const moveTask = async (task: KanbanTask, status: string) => {
    if (task.status === status) return
    setBusyId(task.id)
    try {
      await updateKanbanTask(task.id, {
        status,
        block_reason: status === 'blocked' ? 'Blocked from Desktop board' : undefined
      })
      notify({ message: `Moved to ${statusLabel(status)}`, kind: 'success' })
      refresh()
      if (selectedId === task.id) {
        void qc.invalidateQueries({ queryKey: ['kanban', 'task', task.id] })
      }
    } catch (err) {
      notifyError(err, 'Failed to update task status')
    } finally {
      setBusyId(null)
    }
  }

  if (boardQuery.isLoading && !boardQuery.data) {
    return <PageLoader label="Loading kanban board…" />
  }

  if (boardQuery.isError && !boardQuery.data) {
    return (
      <section className={cn('flex h-full flex-col items-center justify-center gap-3 p-8', className)} {...rest}>
        <p className="max-w-md text-center text-sm text-(--ui-text-secondary)">
          Could not load the kanban board. The backend must expose the kanban plugin API
          (`/api/plugins/kanban/board`).
        </p>
        <Button onClick={() => void boardQuery.refetch()} size="sm" variant="secondary">
          Retry
        </Button>
      </section>
    )
  }

  const selected = detailQuery.data?.task
  const comments = detailQuery.data?.comments ?? []

  return (
    <section className={cn('flex h-full min-h-0 flex-col bg-(--ui-editor-background)', className)} {...rest}>
      <header className="flex shrink-0 flex-wrap items-end gap-2 border-b border-(--ui-border) px-3 py-2">
        <div className="mr-auto min-w-0">
          <h1 className="text-sm font-semibold text-foreground">Kanban</h1>
          <p className="text-[11px] text-(--ui-text-tertiary)">
            Native board · same `/api/plugins/kanban` store as CLI & web
          </p>
        </div>
        <Button onClick={refresh} size="sm" variant="ghost">
          <Codicon className="mr-1" name="refresh" />
          Refresh
        </Button>
      </header>

      <div className="flex shrink-0 flex-wrap items-end gap-2 border-b border-(--ui-border) px-3 py-2">
        <Input
          className="min-w-[12rem] flex-1"
          onChange={e => setTitle(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void onCreate()
            }
          }}
          placeholder="New task title"
          value={title}
        />
        <Input
          className="w-36"
          onChange={e => setAssignee(e.target.value)}
          placeholder="Assignee profile"
          value={assignee}
        />
        <Textarea
          className="min-h-[2.25rem] min-w-[14rem] flex-[1.5] resize-y"
          onChange={e => setBody(e.target.value)}
          placeholder="Optional body"
          rows={1}
          value={body}
        />
        <Button disabled={!title.trim() || creating} onClick={() => void onCreate()} size="sm">
          {creating ? 'Creating…' : 'Add task'}
        </Button>
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-h-0 min-w-0 flex-1 gap-2 overflow-x-auto p-2">
          {KANBAN_COLUMNS.map(col => {
            const tasks = columns[col] ?? []
            return (
              <div
                className="flex w-56 shrink-0 flex-col rounded-md border border-(--ui-border) bg-(--ui-sidebar-surface-background)/40"
                key={col}
              >
                <div className="flex items-center justify-between gap-2 border-b border-(--ui-border) px-2 py-1.5">
                  <span className={cn('text-[11px] font-semibold uppercase tracking-wide', COLUMN_TONE[col])}>
                    {statusLabel(col)}
                  </span>
                  <span className="rounded bg-(--ui-bg-quinary) px-1.5 py-0.5 font-mono text-[10px] text-(--ui-text-tertiary)">
                    {tasks.length}
                  </span>
                </div>
                <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto p-1.5">
                  {tasks.length === 0 ? (
                    <p className="px-1 py-3 text-center text-[11px] text-(--ui-text-tertiary)">Empty</p>
                  ) : (
                    tasks.map(task => (
                      <button
                        className={cn(
                          'rounded border border-transparent bg-(--ui-editor-background) px-2 py-1.5 text-left transition-colors',
                          'hover:border-(--ui-border) hover:bg-(--ui-control-hover-background)',
                          selectedId === task.id && 'border-(--ui-focus-border) bg-(--ui-list-active-selection-background)'
                        )}
                        disabled={busyId === task.id}
                        key={task.id}
                        onClick={() => setSelectedId(task.id)}
                        type="button"
                      >
                        <div className="line-clamp-2 text-xs font-medium text-foreground">{task.title}</div>
                        <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px] text-(--ui-text-tertiary)">
                          {task.assignee ? <span className="font-mono">@{task.assignee}</span> : null}
                          {typeof task.priority === 'number' && task.priority > 0 ? (
                            <span>P{task.priority}</span>
                          ) : null}
                          {task.comment_count ? <span>{task.comment_count}💬</span> : null}
                        </div>
                        {task.latest_summary ? (
                          <p className="mt-1 line-clamp-2 text-[10px] text-(--ui-text-tertiary)">{task.latest_summary}</p>
                        ) : null}
                      </button>
                    ))
                  )}
                </div>
              </div>
            )
          })}
        </div>

        <aside className="flex w-72 shrink-0 flex-col border-l border-(--ui-border) bg-(--ui-sidebar-surface-background)/30">
          {!selectedId ? (
            <div className="flex flex-1 items-center justify-center p-4 text-center text-xs text-(--ui-text-tertiary)">
              Select a task to inspect and move
            </div>
          ) : detailQuery.isLoading && !selected ? (
            <div className="p-4 text-xs text-(--ui-text-tertiary)">Loading…</div>
          ) : selected ? (
            <>
              <div className="border-b border-(--ui-border) px-3 py-2">
                <div className="text-[10px] font-mono text-(--ui-text-tertiary)">{selected.id}</div>
                <h2 className="mt-0.5 text-sm font-semibold text-foreground">{selected.title}</h2>
                <div className="mt-1 text-[11px] text-(--ui-text-secondary)">
                  {statusLabel(selected.status)}
                  {selected.assignee ? ` · @${selected.assignee}` : ''}
                </div>
              </div>
              {selected.body ? (
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap border-b border-(--ui-border) px-3 py-2 text-[11px] text-(--ui-text-secondary)">
                  {selected.body}
                </pre>
              ) : null}
              <div className="border-b border-(--ui-border) px-3 py-2">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)">
                  Move to
                </div>
                <div className="flex flex-wrap gap-1">
                  {KANBAN_COLUMNS.map(col => (
                    <Button
                      disabled={busyId === selected.id || selected.status === col || col === 'running'}
                      key={col}
                      onClick={() => void moveTask(selected, col)}
                      size="sm"
                      variant={selected.status === col ? 'secondary' : 'ghost'}
                    >
                      {statusLabel(col)}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)">
                  Comments
                </div>
                {comments.length === 0 ? (
                  <p className="text-[11px] text-(--ui-text-tertiary)">No comments</p>
                ) : (
                  <ul className="space-y-2">
                    {comments.map((c, i) => (
                      <li className="rounded bg-(--ui-bg-quinary)/50 px-2 py-1.5 text-[11px]" key={c.id ?? i}>
                        <div className="font-mono text-[10px] text-(--ui-text-tertiary)">{c.author || 'unknown'}</div>
                        <div className="whitespace-pre-wrap text-(--ui-text-secondary)">{c.body}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <div className="p-4 text-xs text-red-300/90">Task not found</div>
          )}
        </aside>
      </div>
    </section>
  )
}
