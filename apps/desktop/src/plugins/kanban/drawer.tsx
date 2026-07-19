/**
 * Task drawer — the desktop port of the dashboard's task detail, flat-styled:
 * status menu + meta table, DIAGNOSTICS (the "why is this stuck" panel, with
 * reassign recovery), description (editable), result/summary, dependencies,
 * comments (+composer), activity, run history, and the worker log tail.
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  compactNumber,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  ErrorState,
  host,
  Loader,
  LogView,
  Textarea,
  Tip,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useRef, useState } from 'react'

import {
  $boardSlug,
  addComment,
  deleteTask,
  estimateTask,
  fetchLog,
  fetchProfiles,
  fetchTask,
  logKey,
  patchTask,
  PROFILES_KEY,
  reassignTask,
  reclaimTask,
  taskKey,
  uploadAttachment
} from './api'
import {
  columnMeta,
  COMPLEXITY_LABEL,
  type Diagnostic,
  type DiagnosticAction,
  type KanbanAttachment,
  type KanbanEvent,
  type KanbanTaskDetail,
  SEVERITY_TONE,
  type TaskEstimate
} from './types'
import {
  ago,
  Avatar,
  Callout,
  duration,
  errText,
  isLockedTarget,
  LOCKED_COLUMNS,
  ScrollFade,
  Section,
  shortId,
  StatusMenu,
  useDefaultAssignee
} from './ui'

/**
 * Turn a task_events row into an operator-readable line. The backend logs
 * machine payloads ("status" + {"status":"ready"}); rendering the raw kind
 * made the feed useless ("status · 2 sec. ago" after a drag). Known kinds get
 * prose with the payload folded in; unknown kinds fall back to kind + compact
 * key=value detail so new backend events still say something.
 */
function eventText(event: KanbanEvent): { detail?: string; label: string } {
  let p: Record<string, unknown> = {}

  if (typeof event.payload === 'string' && event.payload) {
    try {
      p = JSON.parse(event.payload) as Record<string, unknown>
    } catch {
      return { label: event.kind.replace(/_/g, ' '), detail: event.payload }
    }
  } else if (event.payload && typeof event.payload === 'object') {
    p = event.payload as Record<string, unknown>
  }

  const str = (key: string): null | string => {
    const value = p[key]

    return typeof value === 'string' && value ? value : null
  }

  const col = (key: string) => {
    const value = str(key)

    return value ? columnMeta(value).label : null
  }

  switch (event.kind) {
    case 'created': {
      const where = col('status')
      const assignee = str('assignee')

      return {
        label: `created${where ? ` in ${where}` : ''}${assignee ? ` · assigned to ${assignee}` : ''}`
      }
    }

    case 'status': {
      const reason = str('reason')

      return {
        label: `moved to ${col('status') ?? '?'}`,
        detail: reason === 'parent_reopened' ? `parent ${str('parent') ?? ''} reopened` : (reason ?? undefined)
      }
    }

    case 'assigned': {
      const assignee = str('assignee')

      return { label: assignee ? `assigned to ${assignee}` : 'unassigned' }
    }

    case 'commented':
      return { label: `comment by ${str('author') ?? 'someone'}` }

    case 'claimed':
      return { label: str('source_status') === 'review' ? 'claimed by a review agent' : 'claimed by a worker' }

    case 'spawned':
      return { label: 'worker started', detail: p.pid != null ? `pid ${p.pid}` : undefined }

    case 'completed':
      return { label: 'completed' }

    case 'blocked':
      return { label: 'blocked — needs human input', detail: str('reason') ?? undefined }

    case 'unblocked':
      return { label: `unblocked${col('status') ? ` → ${col('status')}` : ' → Ready'}` }

    case 'reclaimed':
      return { label: 'reclaimed — returned to the queue', detail: str('reason') ?? undefined }

    case 'specified':
      return { label: 'spec written by the triage agent' }

    case 'promoted':
      return { label: 'dependencies done — promoted to Ready' }

    case 'scheduled':
      return { label: 'scheduled for later', detail: str('reason') ?? undefined }

    case 'archived':
      return { label: 'archived' }

    case 'reprioritized':
      return { label: `priority set to ${p.priority ?? '?'}` }
    default: {
      const detail = Object.entries(p)
        .filter(([, value]) => value != null && typeof value !== 'object')
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(' ')

      return { label: event.kind.replace(/_/g, ' '), detail: detail || undefined }
    }
  }
}

function MetaRow({ children, label }: { children: ReactNode; label: string }) {
  return (
    <>
      <span className="text-(--ui-text-quaternary)">{label}</span>
      <span className="min-w-0 truncate text-(--ui-text-secondary)">{children}</span>
    </>
  )
}

/** The dashboard's diagnostics panel: severity-toned, plain-English, with the
 *  backend's structured recovery actions as buttons. `reassign` is skipped —
 *  the Assignee control in the meta table IS that action, inline. */
function Diagnostics({ items, onReclaim }: { items: Diagnostic[]; onReclaim: () => void }) {
  const act = (action: DiagnosticAction) => {
    if (action.kind === 'reclaim') {
      onReclaim()
    } else if (action.kind === 'cli_hint') {
      void navigator.clipboard.writeText(String(action.payload?.command ?? action.label))
      host.notify({ kind: 'info', message: 'Command copied' })
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {items.map(diag => {
        const tone = SEVERITY_TONE[diag.severity]
        const actions = diag.actions.filter(action => action.kind === 'reclaim' || action.kind === 'cli_hint')

        return (
          <Callout
            key={`${diag.kind}-${diag.last_seen_at}`}
            title={`${diag.title}${diag.count > 1 ? ` ×${diag.count}` : ''}`}
            tone={tone}
          >
            <p className="whitespace-pre-wrap text-[0.71rem] leading-relaxed text-(--ui-text-secondary)">
              {diag.detail}
            </p>
            {actions.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {actions.map(action => (
                  <Button
                    key={`${action.kind}-${action.label}`}
                    onClick={() => act(action)}
                    size="xs"
                    variant={action.suggested ? 'secondary' : 'outline'}
                  >
                    {action.kind === 'cli_hint' && <Codicon name="copy" size="0.7rem" />}
                    {action.label}
                  </Button>
                ))}
              </div>
            )}
          </Callout>
        )
      })}
    </div>
  )
}

/** Jira-style inline assignee editor: the meta row IS the control — click the
 *  assignee to reassign (reclaims a running worker first, resets the failure
 *  streak — the explicit human recovery action). */
function AssigneeMenu({
  current,
  onReassign
}: {
  current: null | string | undefined
  onReassign: (p: string) => void
}) {
  const { data: roster } = useQuery({ queryKey: PROFILES_KEY, queryFn: fetchProfiles, staleTime: 60_000 })

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className="-mx-1 inline-flex max-w-full items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-(--chrome-action-hover)"
          type="button"
        >
          {current ? (
            <>
              <Avatar name={current} size="0.875rem" />
              <span className="truncate">{current}</span>
            </>
          ) : (
            <span className="text-(--ui-text-quaternary)">unassigned</span>
          )}
          <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="chevron-down" size="0.65rem" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {(roster?.profiles ?? []).map(profile => (
          <DropdownMenuItem key={profile.name} onSelect={() => onReassign(profile.name)}>
            <Avatar name={profile.name} size="0.875rem" />
            {profile.name}
            {profile.name === current && <Codicon className="ml-auto" name="check" size="0.8rem" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// Mirrors the review pane's commit-message field: one row tall to start
// (button-height), CSS field-sizing grows it with content, button hugs the
// bottom edge as it grows.
function CommentComposer({ onSubmit, pending }: { onSubmit: (body: string) => void; pending: boolean }) {
  const [body, setBody] = useState('')

  const submit = () => {
    const trimmed = body.trim()

    if (trimmed && !pending) {
      onSubmit(trimmed)
      setBody('')
    }
  }

  return (
    <div className="relative">
      <Textarea
        className="field-sizing-content max-h-40 min-h-0 resize-none pr-[5rem]"
        onChange={event => setBody(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
        placeholder="Add a comment…"
        rows={1}
        size="sm"
        value={body}
      />
      <Button
        className="absolute top-1 right-1"
        disabled={!body.trim() || pending}
        onClick={submit}
        size="xs"
        variant="secondary"
      >
        Comment
      </Button>
    </div>
  )
}

function DescriptionSection({ body, onSave }: { body: null | string | undefined; onSave: (body: string) => void }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  return (
    <Section
      action={
        <Button
          aria-label={editing ? 'Cancel edit' : 'Edit description'}
          onClick={() => {
            setDraft(body ?? '')
            setEditing(!editing)
          }}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name={editing ? 'close' : 'edit'} size="0.75rem" />
        </Button>
      }
      label="Description"
    >
      {editing ? (
        <div className="flex flex-col gap-1.5">
          <Textarea
            className="min-h-24 text-[0.75rem]"
            onChange={event => setDraft(event.target.value)}
            value={draft}
          />
          <Button
            className="self-end"
            onClick={() => {
              onSave(draft)
              setEditing(false)
            }}
            size="xs"
            variant="secondary"
          >
            Save
          </Button>
        </div>
      ) : body ? (
        <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{body}</p>
      ) : (
        <p className="text-[0.8125rem] text-(--ui-text-quaternary)">No description yet.</p>
      )}
    </Section>
  )
}

// `latest_summary` is just the newest non-null run summary. A reclaim writes an
// administrative note into that slot; hide those (Runs still shows them).
const isAdminSummary = (summary: string) => /^status changed to \w+ \(dashboard\/direct\)$/.test(summary)

function AttachmentsSection({
  attachments,
  onUpload,
  pending
}: {
  attachments: KanbanAttachment[]
  onUpload: (file: File) => void
  pending: boolean
}) {
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <Section
      action={
        <>
          <input
            hidden
            onChange={event => {
              const file = event.target.files?.[0]

              if (file) {
                onUpload(file)
              }

              event.target.value = ''
            }}
            ref={fileRef}
            type="file"
          />
          <Button
            aria-label="Upload attachment"
            disabled={pending}
            onClick={() => fileRef.current?.click()}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name={pending ? 'sync' : 'cloud-upload'} size="0.8rem" spinning={pending} />
          </Button>
        </>
      }
      label={`Attachments · ${attachments.length}`}
    >
      {attachments.length > 0 ? (
        <ul className="flex flex-col gap-1">
          {attachments.map(attachment => (
            <li className="flex items-center gap-1.5 text-[0.75rem] text-(--ui-text-tertiary)" key={attachment.id}>
              <Codicon name="file" size="0.75rem" />
              {attachment.filename}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[0.75rem] text-(--ui-text-quaternary)">No attachments yet.</p>
      )}
    </Section>
  )
}

// Rough effort estimate via the auxiliary (auto-routed) model. Tokens +
// complexity, never dollars — providers don't report cost reliably. Gated
// behind an explicit click + disclaimer since it makes a model call. The
// control keeps a stable footprint (spinner swaps in place) so nothing jumps.
function EstimateSection({ id }: { id: string }) {
  const [result, setResult] = useState<null | TaskEstimate>(null)

  const est = useMutation({
    mutationFn: () => estimateTask(id),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: r => {
      if (r.ok) {
        setResult(r)
      } else {
        host.notify({ kind: 'warning', message: r.reason || 'Could not estimate' })
      }
    }
  })

  // A new task resets the cached estimate (the drawer reuses one instance).
  useEffect(() => setResult(null), [id])

  return (
    <Section label="Estimate">
      {result?.ok ? (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[0.8125rem]">
            <span className="font-medium tabular-nums text-(--ui-text-secondary)">
              ~{compactNumber(result.est_tokens)} tok
            </span>
            {result.complexity && (
              <span className="text-(--ui-text-tertiary)">
                · {COMPLEXITY_LABEL[result.complexity] ?? result.complexity}
              </span>
            )}
            <Tip label="Re-estimate">
              <Button
                aria-label="Re-estimate"
                className="ml-auto"
                disabled={est.isPending}
                onClick={() => est.mutate()}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="refresh" size="0.75rem" spinning={est.isPending} />
              </Button>
            </Tip>
          </div>
          {result.rationale && (
            <p className="text-[0.6875rem] leading-relaxed text-(--ui-text-quaternary)">{result.rationale}</p>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <Button disabled={est.isPending} onClick={() => est.mutate()} size="xs" variant="outline">
            <Codicon name={est.isPending ? 'loading' : 'dashboard'} size="0.75rem" spinning={est.isPending} />
            {est.isPending ? 'Estimating…' : 'Estimate effort'}
          </Button>
          <Tip label="Runs a quick auxiliary-model call to estimate tokens + complexity. A rough guide, not a bill.">
            <span className="text-[0.625rem] text-(--ui-text-quaternary)">makes a model call</span>
          </Tip>
        </div>
      )}
    </Section>
  )
}

export function TaskDrawer({
  columns,
  id,
  onClose,
  onOpen
}: {
  columns: string[]
  id: null | string
  onClose: () => void
  onOpen: (id: string) => void
}) {
  const qc = useQueryClient()
  const slug = useValue($boardSlug)

  // Socket-invalidated (bindApi); the interval is only the socketless heartbeat.
  const { data: detail, error } = useQuery({
    enabled: !!id,
    queryFn: () => fetchTask(id!),
    queryKey: taskKey(slug, id ?? ''),
    refetchInterval: 30_000
  })

  const task = detail?.task
  const running = task?.status === 'running'
  const defaultAssignee = useDefaultAssignee()

  const { data: log } = useQuery({
    enabled: !!id,
    queryFn: () => fetchLog(id!),
    queryKey: logKey(slug, id ?? ''),
    refetchInterval: running ? 3_000 : 15_000
  })

  // Esc closes the drawer even though it isn't modal (no backdrop to click off).
  useEffect(() => {
    if (!id) {
      return
    }

    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [id, onClose])

  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: taskKey(slug, id!) })
    void qc.invalidateQueries({ queryKey: ['kanban', 'board', slug] })
  }

  // Optimistic status change against the task cache; rolls back + toasts on a
  // rejected transition (the backend enforces the workflow).
  const moveMut = useMutation({
    mutationFn: (status: string) => patchTask(id!, { status }),
    onMutate: async status => {
      await qc.cancelQueries({ queryKey: taskKey(slug, id!) })
      const previous = qc.getQueryData<KanbanTaskDetail>(taskKey(slug, id!))

      if (previous) {
        qc.setQueryData(taskKey(slug, id!), { ...previous, task: { ...previous.task, status } })
      }

      return { previous }
    },
    onError: (err, _status, context) => {
      if (context?.previous) {
        qc.setQueryData(taskKey(slug, id!), context.previous)
      }

      host.notify({ kind: 'error', message: errText(err) })
    },
    onSettled: invalidate
  })

  const mutate = (fn: () => Promise<unknown>, onDone?: () => void) => () =>
    fn().then(
      () => {
        invalidate()
        onDone?.()
      },
      (err: unknown) => host.notify({ kind: 'error', message: errText(err) })
    )

  const commentMut = useMutation({
    mutationFn: (body: string) => addComment(id!, body),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: invalidate
  })

  const uploadMut = useMutation({
    mutationFn: async (file: File) =>
      uploadAttachment(id!, {
        bytes: await file.arrayBuffer(),
        contentType: file.type || undefined,
        filename: file.name
      }),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: invalidate
  })

  if (!id) {
    return null
  }

  const errorMessage = error ? errText(error) : null

  const move = (status: string) => {
    if (!task || status === task.status) {
      return
    }

    if (isLockedTarget(status)) {
      host.notify({ kind: 'info', message: LOCKED_COLUMNS[status] })

      return
    }

    moveMut.mutate(status)
  }

  return (
    <div className="absolute inset-y-0 right-0 z-20 flex w-[26rem] flex-col border-l border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) duration-150 ease-out animate-in fade-in slide-in-from-right-4">
      <header className="flex flex-col gap-2 px-4 pt-3.5 pb-3">
        <div className="flex items-center gap-2">
          {task ? (
            <StatusMenu columns={columns} onMove={move} status={task.status} />
          ) : (
            <span className="font-mono text-sm text-(--ui-text-tertiary)">{shortId(id)}</span>
          )}
          {task && (
            <span className="font-mono text-[0.625rem] text-(--ui-text-quaternary)" data-selectable-text="true">
              {shortId(task.id)}
            </span>
          )}
          <div className="ml-auto flex items-center gap-0.5">
            {task && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    aria-label="Task actions"
                    className="grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
                    type="button"
                  >
                    <Codicon name="ellipsis" size="0.9rem" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onSelect={() => {
                      void navigator.clipboard.writeText(task.id)
                      host.notify({ kind: 'info', message: `Copied ${task.id}` })
                    }}
                  >
                    <Codicon name="copy" size="0.85rem" />
                    Copy task id
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => {
                      void navigator.clipboard.writeText(task.title || task.id)
                      host.notify({ kind: 'info', message: 'Copied title' })
                    }}
                  >
                    <Codicon name="copy" size="0.85rem" />
                    Copy title
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={mutate(() => patchTask(task.id, { status: 'archived' }), onClose)}>
                    <Codicon name="archive" size="0.85rem" />
                    Archive task
                  </DropdownMenuItem>
                  <DropdownMenuItem className="text-destructive" onSelect={mutate(() => deleteTask(task.id), onClose)}>
                    <Codicon name="trash" size="0.85rem" />
                    Delete task
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            <button
              aria-label="Close"
              className="grid size-6 place-items-center rounded text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
              onClick={onClose}
              type="button"
            >
              <Codicon name="close" size="0.9rem" />
            </button>
          </div>
        </div>
        {task && (
          <h2 className="text-sm leading-snug font-semibold text-foreground" data-selectable-text="true">
            {task.title || task.id}
          </h2>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4" data-selectable-text="true">
        {errorMessage ? (
          <ErrorState title={errorMessage} />
        ) : !detail || !task ? (
          <div className="grid h-32 place-items-center">
            <Loader type="lemniscate-bloom" />
          </div>
        ) : (
          <div className="flex flex-col gap-4 text-sm">
            <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-x-3 gap-y-1 text-[0.71rem]">
              <MetaRow label="Assignee">
                <AssigneeMenu
                  current={task.assignee}
                  onReassign={profile => void mutate(() => reassignTask(task.id, profile))()}
                />
              </MetaRow>
              {typeof task.priority === 'number' && <MetaRow label="Priority">{task.priority}</MetaRow>}
              {task.tenant && <MetaRow label="Tenant">{task.tenant}</MetaRow>}
              {task.workspace_path && (
                <MetaRow label="Workspace">
                  {task.workspace_kind ? `${task.workspace_kind}: ` : ''}
                  {task.workspace_path}
                </MetaRow>
              )}
              {task.created_by && <MetaRow label="Created by">{task.created_by}</MetaRow>}
              {ago(task.created_at) && <MetaRow label="Created">{ago(task.created_at)}</MetaRow>}
              {running && task.worker_pid ? <MetaRow label="Worker pid">{task.worker_pid}</MetaRow> : null}
            </div>

            {task.status === 'ready' && !task.assignee && !defaultAssignee && (
              <Callout title="Ready, but unassigned — this card will never run." tone={SEVERITY_TONE.warning}>
                <p className="text-[0.71rem] leading-relaxed text-(--ui-text-secondary)">
                  The dispatcher only claims Ready cards that have an assignee. Pick a profile in the Assignee field
                  above (or set a default assignee in the orchestration settings) and it runs within a minute.
                </p>
              </Callout>
            )}

            {task.diagnostics && task.diagnostics.length > 0 && (
              <Section label={`Diagnostics · ${task.diagnostics.length}`}>
                <Diagnostics items={task.diagnostics} onReclaim={() => void mutate(() => reclaimTask(task.id))()} />
              </Section>
            )}

            <DescriptionSection body={task.body} onSave={body => void mutate(() => patchTask(task.id, { body }))()} />

            <EstimateSection id={task.id} />

            {task.result && (
              <Section label="Result">
                <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{task.result}</p>
              </Section>
            )}

            {task.latest_summary && !isAdminSummary(task.latest_summary) && (
              <Section label="Latest summary">
                <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{task.latest_summary}</p>
              </Section>
            )}

            {(detail.links.parents.length > 0 || detail.links.children.length > 0) && (
              <Section label="Dependencies">
                {(['parents', 'children'] as const).map(side =>
                  detail.links[side].length > 0 ? (
                    <div className="flex flex-wrap items-center gap-1.5" key={side}>
                      <span className="text-[0.6875rem] text-(--ui-text-quaternary)">
                        {side === 'parents' ? 'Blocked by' : 'Blocks'}
                      </span>
                      {detail.links[side].map(linked => (
                        <button
                          className="rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5 font-mono text-[0.625rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
                          key={linked}
                          onClick={() => onOpen(linked)}
                          type="button"
                        >
                          {shortId(linked)}
                        </button>
                      ))}
                    </div>
                  ) : null
                )}
              </Section>
            )}

            <Section label={`Comments · ${detail.comments.length}`}>
              {detail.comments.length > 0 && (
                <ul className="flex flex-col gap-2">
                  {detail.comments.map(comment => (
                    <li className="text-[0.75rem]" key={comment.id}>
                      <span className="font-medium text-(--ui-text-secondary)">{comment.author}</span>
                      <span className="ml-2 text-[0.625rem] text-(--ui-text-quaternary)">
                        {ago(comment.created_at)}
                      </span>
                      <p className="whitespace-pre-wrap text-(--ui-text-tertiary)">{comment.body}</p>
                    </li>
                  ))}
                </ul>
              )}
              <CommentComposer onSubmit={body => commentMut.mutate(body)} pending={commentMut.isPending} />
            </Section>

            {detail.events.length > 0 && (
              <Section label={`Activity · ${detail.events.length}`}>
                <ScrollFade deps={detail.events.length} max="7rem">
                  <ul className="flex flex-col gap-1">
                    {detail.events.map(event => {
                      const { detail: extra, label } = eventText(event)

                      return (
                        <li className="flex items-baseline gap-2 text-[0.6875rem]" key={event.id}>
                          <span className="shrink-0 text-(--ui-text-secondary)">{label}</span>
                          {extra && (
                            <span
                              className="min-w-0 truncate text-[0.625rem] text-(--ui-text-quaternary)"
                              title={extra}
                            >
                              {extra}
                            </span>
                          )}
                          <span className="ml-auto shrink-0 text-(--ui-text-quaternary)">{ago(event.created_at)}</span>
                        </li>
                      )
                    })}
                  </ul>
                </ScrollFade>
              </Section>
            )}

            {detail.runs.length > 0 && (
              <Section label={`Runs · ${detail.runs.length}`}>
                <ScrollFade max="11rem">
                  <ul className="flex flex-col gap-1.5">
                    {detail.runs.map(run => {
                      const failed = ['crashed', 'failed', 'timed_out', 'gave_up'].includes(run.outcome ?? run.status)

                      return (
                        <li className="flex flex-col gap-0.5 text-[0.71rem]" key={run.id}>
                          <div className="flex items-center gap-2">
                            <Badge size="xs" variant={failed ? 'destructive' : 'muted'}>
                              {run.outcome ?? run.status}
                            </Badge>
                            {run.profile && <span className="text-(--ui-text-tertiary)">{run.profile}</span>}
                            {duration(run.started_at, run.ended_at) && (
                              <span className="text-(--ui-text-quaternary)">
                                {duration(run.started_at, run.ended_at)}
                              </span>
                            )}
                            <span className="ml-auto shrink-0 text-(--ui-text-quaternary)">
                              {ago(run.ended_at ?? run.started_at)}
                            </span>
                          </div>
                          {(run.error || run.summary) && (
                            <p
                              className={cn(
                                'line-clamp-2 whitespace-pre-wrap',
                                run.error ? 'text-destructive' : 'text-(--ui-text-quaternary)'
                              )}
                            >
                              {run.error ?? run.summary}
                            </p>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                </ScrollFade>
              </Section>
            )}

            {log?.exists && log.content && (
              <Section label={`Worker log${log.truncated ? ' · tail' : ''}`}>
                <ScrollFade deps={log.content.length} max="12rem">
                  <LogView className="border-0 px-0">{log.content}</LogView>
                </ScrollFade>
              </Section>
            )}

            <AttachmentsSection
              attachments={detail.attachments}
              onUpload={file => uploadMut.mutate(file)}
              pending={uploadMut.isPending}
            />
          </div>
        )}
      </div>
    </div>
  )
}
