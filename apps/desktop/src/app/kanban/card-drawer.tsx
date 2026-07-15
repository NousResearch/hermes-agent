import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { type Translations, useI18n } from '@/i18n'
import { addKanbanComment, getKanbanAssignees, getKanbanTask, getKanbanTaskLog, updateKanbanTask } from '@/kanban-api'
import { cn } from '@/lib/utils'
import {
  $kanbanActiveBoard,
  $kanbanBoard,
  allowedDropColumns,
  applyKanbanTaskPatch,
  refreshKanbanBoard,
  setKanbanSelectedTaskId
} from '@/store/kanban'
import { notify, notifyError } from '@/store/notifications'
import type { KanbanRun, KanbanTaskDetail, KanbanUpdateTaskPayload } from '@/types/kanban'

import { PanelBlock, PanelPill, type PanelPillTone, PanelSectionLabel } from '../overlays/panel'

// Sentinel for the Radix Select "unassigned" row (empty item values are not
// allowed); mapped to `assignee: ''` on PATCH, which unassigns server-side.
const UNASSIGNED = '__unassigned__'

const RUN_TONE: Record<string, PanelPillTone> = {
  running: 'good',
  done: 'good',
  blocked: 'warn',
  released: 'muted',
  reclaimed: 'muted',
  crashed: 'bad',
  failed: 'bad',
  timed_out: 'bad'
}

function formatEpoch(seconds?: null | number): string {
  if (!seconds) {
    return '—'
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.valueOf()) ? '—' : date.toLocaleString()
}

function formatBytes(size?: null | number): string {
  if (size === null || size === undefined) {
    return '—'
  }

  if (size < 1024) {
    return `${size} B`
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

interface KanbanCardDrawerProps {
  onClose: () => void
  taskId: string
}

export function KanbanCardDrawer({ onClose, taskId }: KanbanCardDrawerProps) {
  const { t } = useI18n()
  const k = t.kanban
  const board = useStore($kanbanBoard)
  const activeBoard = useStore($kanbanActiveBoard)

  const [detail, setDetail] = useState<KanbanTaskDetail | null>(null)
  const [loadError, setLoadError] = useState<null | string>(null)
  const [assignees, setAssignees] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [commentDraft, setCommentDraft] = useState('')
  const [sendingComment, setSendingComment] = useState(false)

  const reload = useCallback(async () => {
    try {
      setDetail(await getKanbanTask(taskId, $kanbanActiveBoard.get()))
      setLoadError(null)
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error))
    }
  }, [taskId])

  useEffect(() => {
    void reload()
    // Re-pull the detail whenever the board cursor moves — comments/events/runs
    // append task_events rows, so the cursor is a reliable staleness signal.
  }, [reload, board?.latest_event_id])

  useEffect(() => {
    let cancelled = false

    getKanbanAssignees(activeBoard)
      .then(list => {
        if (!cancelled) {
          setAssignees(list.map(entry => entry.name))
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [activeBoard])

  const patchTask = useCallback(
    async (updates: KanbanUpdateTaskPayload) => {
      setSaving(true)

      try {
        const response = await updateKanbanTask(taskId, updates, $kanbanActiveBoard.get())

        // Server-authoritative: render whatever came back (a status PATCH may
        // land on a different column than requested under local board rules).
        if (response.task) {
          applyKanbanTaskPatch(response.task)

          if (updates.status && response.task.status !== updates.status) {
            notify({
              kind: 'info',
              message: k.movedByRules(k.columnLabels[response.task.status] ?? response.task.status)
            })
          }
        }

        await reload()
        void refreshKanbanBoard()
      } catch (error) {
        notifyError(error, k.updateFailed)
        // Refetch so the editors snap back to the server's state.
        await reload()
      } finally {
        setSaving(false)
      }
    },
    [k, reload, taskId]
  )

  async function submitComment() {
    const body = commentDraft.trim()

    if (!body || sendingComment) {
      return
    }

    setSendingComment(true)

    try {
      await addKanbanComment(taskId, body, $kanbanActiveBoard.get())
      setCommentDraft('')
      await reload()
      void refreshKanbanBoard()
    } catch (error) {
      notifyError(error, k.commentFailed)
    } finally {
      setSendingComment(false)
    }
  }

  const task = detail?.task
  const statusOptions = task ? [task.status, ...allowedDropColumns(task.status)] : []

  const assigneeOptions =
    task?.assignee && !assignees.includes(task.assignee) ? [task.assignee, ...assignees] : assignees

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background)">
      <header className="flex shrink-0 items-start justify-between gap-2 border-b border-(--ui-stroke-secondary) px-3 py-2.5">
        <div className="min-w-0">
          <h3 className="break-words text-[0.85rem] font-semibold leading-snug text-foreground">
            {task?.title ?? taskId}
          </h3>
          <p className="font-mono text-[0.62rem] text-muted-foreground/55">{taskId}</p>
        </div>
        <Button
          aria-label={t.common.close}
          className="shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
          onClick={onClose}
          size="icon"
          variant="ghost"
        >
          <Codicon name="close" size="0.875rem" />
        </Button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain px-3 py-3">
        {!detail && !loadError ? (
          <div className="flex items-center gap-1.5 py-2 text-xs text-muted-foreground">
            <Codicon name="loading" size="0.75rem" spinning />
            {k.loadingDetail}
          </div>
        ) : loadError && !detail ? (
          <div className="rounded bg-destructive/10 p-2 text-xs text-destructive">{loadError}</div>
        ) : task ? (
          <>
            <section className="grid grid-cols-1 gap-2">
              <DrawerField label={k.status}>
                <Select
                  disabled={saving}
                  onValueChange={value => void patchTask({ status: value })}
                  value={task.status}
                >
                  <SelectTrigger className="h-7 rounded-md text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {statusOptions.map(status => (
                      <SelectItem key={status} value={status}>
                        {k.columnLabels[status] ?? status}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </DrawerField>

              <DrawerField label={k.assignee}>
                <Select
                  disabled={saving}
                  onValueChange={value => void patchTask({ assignee: value === UNASSIGNED ? '' : value })}
                  value={task.assignee ?? UNASSIGNED}
                >
                  <SelectTrigger className="h-7 rounded-md text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={UNASSIGNED}>{k.unassigned}</SelectItem>
                    {assigneeOptions.map(name => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </DrawerField>

              <DrawerField label={k.priority}>
                <Input
                  className="h-7 text-xs tabular-nums"
                  defaultValue={String(task.priority)}
                  disabled={saving}
                  key={`priority-${task.priority}`}
                  onBlur={event => {
                    const next = Number.parseInt(event.target.value, 10)

                    if (Number.isInteger(next) && next !== task.priority) {
                      void patchTask({ priority: next })
                    }
                  }}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {
                      event.currentTarget.blur()
                    }
                  }}
                  type="number"
                />
              </DrawerField>
            </section>

            <section className="space-y-1.5">
              <PanelSectionLabel>{k.description}</PanelSectionLabel>
              {task.body?.trim() ? (
                <PanelBlock className="max-h-56">{task.body}</PanelBlock>
              ) : (
                <p className="text-xs text-muted-foreground/60">{k.noDescription}</p>
              )}
            </section>

            {task.latest_summary?.trim() ? (
              <section className="space-y-1.5">
                <PanelSectionLabel>{k.latestSummary}</PanelSectionLabel>
                <PanelBlock className="max-h-56">{task.latest_summary}</PanelBlock>
              </section>
            ) : null}

            {(task.diagnostics?.length ?? 0) > 0 && (
              <section className="space-y-1.5">
                <PanelSectionLabel>{k.diagnostics}</PanelSectionLabel>
                <div className="space-y-1">
                  {task.diagnostics?.map(diag => (
                    <div
                      className={cn(
                        'rounded p-2 text-[0.68rem] leading-relaxed',
                        diag.severity === 'warning'
                          ? 'bg-amber-500/10 text-amber-600 dark:text-amber-300'
                          : 'bg-destructive/10 text-destructive'
                      )}
                      key={`${diag.kind}-${diag.last_seen_at}`}
                    >
                      <div className="font-medium">{diag.title}</div>
                      <div className="opacity-80">{diag.detail}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="space-y-1.5">
              <PanelSectionLabel>{k.dependencies}</PanelSectionLabel>
              <LinkChips ids={detail.links.parents} k={k} label={k.parents} />
              <LinkChips ids={detail.links.children} k={k} label={k.children} />
            </section>

            <section className="space-y-1.5">
              <PanelSectionLabel>
                {k.comments}
                {detail.comments.length > 0 ? ` · ${detail.comments.length}` : ''}
              </PanelSectionLabel>
              {detail.comments.length === 0 ? (
                <p className="text-xs text-muted-foreground/60">{k.noComments}</p>
              ) : (
                <div className="space-y-1.5">
                  {detail.comments.map(comment => (
                    <div className="rounded bg-foreground/5 p-2" key={comment.id}>
                      <div className="mb-0.5 flex items-baseline justify-between gap-2 text-[0.62rem] text-muted-foreground/60">
                        <span className="truncate font-medium">{comment.author ?? '—'}</span>
                        <span className="shrink-0 tabular-nums">{formatEpoch(comment.created_at)}</span>
                      </div>
                      <div className="whitespace-pre-wrap break-words text-[0.7rem] leading-relaxed text-foreground/85">
                        {comment.body}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex flex-col gap-1.5">
                <Textarea
                  className="min-h-14 text-xs"
                  onChange={event => setCommentDraft(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                      event.preventDefault()
                      void submitComment()
                    }
                  }}
                  placeholder={k.commentPlaceholder}
                  value={commentDraft}
                />
                <Button
                  className="self-end"
                  disabled={sendingComment || !commentDraft.trim()}
                  onClick={() => void submitComment()}
                  size="sm"
                  variant="outline"
                >
                  {sendingComment ? t.common.saving : k.addComment}
                </Button>
              </div>
            </section>

            <section className="space-y-1.5">
              <PanelSectionLabel>
                {k.runHistory}
                {detail.runs.length > 0 ? ` · ${detail.runs.length}` : ''}
              </PanelSectionLabel>
              {detail.runs.length === 0 ? (
                <p className="text-xs text-muted-foreground/60">{k.noRuns}</p>
              ) : (
                <div className="space-y-1">
                  {detail.runs.map(run => (
                    <RunRow k={k} key={run.id} run={run} />
                  ))}
                </div>
              )}
              <WorkerLogSection k={k} taskId={taskId} />
            </section>

            <section className="space-y-1.5">
              <PanelSectionLabel>
                {k.attachments}
                {detail.attachments.length > 0 ? ` · ${detail.attachments.length}` : ''}
              </PanelSectionLabel>
              {detail.attachments.length === 0 ? (
                <p className="text-xs text-muted-foreground/60">{k.noAttachments}</p>
              ) : (
                <div className="space-y-px">
                  {detail.attachments.map(attachment => (
                    <div
                      className="flex items-center justify-between gap-2 rounded px-1.5 py-1 text-[0.7rem] text-foreground/85"
                      key={attachment.id}
                    >
                      <span className="flex min-w-0 items-center gap-1.5">
                        <Codicon className="shrink-0 text-muted-foreground/55" name="file" size="0.75rem" />
                        <span className="truncate">{attachment.filename}</span>
                      </span>
                      <span className="shrink-0 text-[0.62rem] tabular-nums text-muted-foreground/50">
                        {formatBytes(attachment.size)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="space-y-1.5">
              <PanelSectionLabel>
                {k.events}
                {detail.events.length > 0 ? ` · ${detail.events.length}` : ''}
              </PanelSectionLabel>
              {detail.events.length === 0 ? (
                <p className="text-xs text-muted-foreground/60">{k.noEvents}</p>
              ) : (
                <div className="space-y-px">
                  {[...detail.events].reverse().map(event => (
                    <div
                      className="flex items-baseline justify-between gap-2 rounded px-1.5 py-0.5 text-[0.66rem]"
                      key={event.id}
                    >
                      <span className="min-w-0">
                        <span className="font-medium text-foreground/80">{event.kind}</span>
                        {event.payload ? (
                          <span className="ml-1.5 break-all text-muted-foreground/60">
                            {JSON.stringify(event.payload)}
                          </span>
                        ) : null}
                      </span>
                      <span className="shrink-0 tabular-nums text-muted-foreground/45">
                        {formatEpoch(event.created_at)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        ) : null}
      </div>
    </aside>
  )
}

function DrawerField({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <div className="grid grid-cols-[4.5rem_1fr] items-center gap-2">
      <span className="text-[0.68rem] text-muted-foreground/60">{label}</span>
      {children}
    </div>
  )
}

function LinkChips({ ids, k, label }: { ids: string[]; k: Translations['kanban']; label: string }) {
  return (
    <div className="flex flex-wrap items-center gap-1 text-[0.66rem]">
      <span className="text-muted-foreground/55">{label}</span>
      {ids.length === 0 ? (
        <span className="text-muted-foreground/40">{k.none}</span>
      ) : (
        ids.map(id => (
          <button
            className="rounded-full bg-foreground/8 px-1.5 py-px font-mono text-[0.62rem] text-foreground/75 transition-colors hover:bg-foreground/15 hover:text-foreground"
            key={id}
            onClick={() => setKanbanSelectedTaskId(id)}
            type="button"
          >
            {id}
          </button>
        ))
      )}
    </div>
  )
}

function RunRow({ k, run }: { k: Translations['kanban']; run: KanbanRun }) {
  const state = run.outcome ?? run.status

  return (
    <div className="rounded bg-foreground/5 p-2 text-[0.66rem]">
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="shrink-0 font-mono text-muted-foreground/55">#{run.id}</span>
          <span className="truncate text-foreground/80">{run.profile ?? '—'}</span>
        </span>
        <PanelPill tone={RUN_TONE[state] ?? 'muted'}>{state}</PanelPill>
      </div>
      <div className="mt-0.5 flex items-baseline justify-between gap-2 tabular-nums text-muted-foreground/50">
        <span>{formatEpoch(run.started_at)}</span>
        <span>{run.ended_at ? formatEpoch(run.ended_at) : k.runActive}</span>
      </div>
      {run.summary?.trim() ? (
        <div className="mt-1 line-clamp-3 whitespace-pre-wrap break-words leading-relaxed text-foreground/75">
          {run.summary}
        </div>
      ) : null}
      {run.error?.trim() ? <div className="mt-1 break-words text-destructive/85">{run.error}</div> : null}
    </div>
  )
}

function WorkerLogSection({ k, taskId }: { k: Translations['kanban']; taskId: string }) {
  const [open, setOpen] = useState(false)
  const [log, setLog] = useState<null | string>(null)
  const [loading, setLoading] = useState(false)

  async function toggle() {
    if (open) {
      setOpen(false)

      return
    }

    setOpen(true)
    setLoading(true)

    try {
      const result = await getKanbanTaskLog(taskId, $kanbanActiveBoard.get())

      setLog(result.exists ? result.content : null)
    } catch {
      setLog(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-1">
      <button
        className="flex items-center gap-1 text-[0.66rem] text-muted-foreground/70 transition-colors hover:text-foreground"
        onClick={() => void toggle()}
        type="button"
      >
        <Codicon name={open ? 'chevron-down' : 'chevron-right'} size="0.7rem" />
        {k.workerLog}
      </button>
      {open ? (
        loading ? (
          <div className="flex items-center gap-1.5 py-1 text-xs text-muted-foreground">
            <Codicon name="loading" size="0.75rem" spinning />
            {k.loadingLog}
          </div>
        ) : log ? (
          <PanelBlock className="max-h-64 font-mono">{log}</PanelBlock>
        ) : (
          <p className="text-xs text-muted-foreground/60">{k.noWorkerLog}</p>
        )
      ) : null}
    </div>
  )
}
