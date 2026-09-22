/**
 * Task drawer — the desktop port of the dashboard's task detail, flat-styled:
 * status menu + meta table, DIAGNOSTICS (the "why is this stuck" panel, with
 * reassign recovery), description (editable), result/summary, dependencies,
 * comments (+composer), activity folded by attempt, and the worker log tail.
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
  isSubmitEnter,
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
  boardKeyPrefix,
  deleteTask,
  estimateTask,
  fetchLog,
  fetchProfiles,
  fetchTask,
  logKey,
  patchTask,
  profilesKey,
  reassignTask,
  reclaimTask,
  routedToScope,
  taskKey,
  uploadAttachment,
  useKanbanScope
} from './api'
import { ModelOverrideField, overridePatch } from './model-override'
import {
  type Diagnostic,
  type DiagnosticAction,
  type KanbanAttachment,
  type KanbanEvent,
  type KanbanRun,
  type KanbanTaskDetail,
  SEVERITY_TONE,
  type TaskEstimate
} from './types'
import {
  ago,
  Avatar,
  Callout,
  columnLabel,
  DEP_TONES,
  depSegment,
  duration,
  errText,
  FIELD_LABEL,
  isLockedTarget,
  type KanbanText,
  lockedReason,
  ScrollFade,
  Section,
  shortId,
  StatusMenu,
  useDefaultAssignee,
  useKanban
} from './ui'

/**
 * Turn a task_events row into an operator-readable line. The backend logs
 * machine payloads ("status" + {"status":"ready"}); rendering the raw kind
 * made the feed useless ("status · 2 sec. ago" after a drag). Known kinds get
 * prose with the payload folded in; unknown kinds fall back to kind + compact
 * key=value detail so new backend events still say something.
 */
function eventText(event: KanbanEvent, k: KanbanText): { detail?: string; label: string } {
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

    return value ? columnLabel(k, value) : null
  }

  switch (event.kind) {
    case 'created':
      return { label: k.evtCreated(col('status') ?? '', str('assignee') ?? '') }
    case 'status': {
      const reason = str('reason')

      return {
        label: k.evtMovedTo(col('status') ?? '?'),
        detail: reason === 'parent_reopened' ? k.evtParentReopened(str('parent') ?? '') : (reason ?? undefined)
      }
    }

    case 'assigned': {
      const assignee = str('assignee')

      return { label: assignee ? k.evtAssignedTo(assignee) : k.evtUnassigned }
    }

    case 'commented':
      return { label: k.evtCommentBy(str('author') ?? k.someone) }

    case 'claimed':
      return { label: str('source_status') === 'review' ? k.evtClaimedReview : k.evtClaimedWorker }

    case 'spawned':
      return { label: k.evtWorkerStarted, detail: p.pid != null ? `pid ${p.pid}` : undefined }

    case 'completed':
      return { label: k.evtCompleted }

    case 'blocked':
      return { label: k.evtBlocked, detail: str('reason') ?? undefined }

    case 'unblocked':
      return { label: k.evtUnblocked(col('status') ?? '') }

    case 'reclaimed':
      return { label: k.evtReclaimed, detail: str('reason') ?? undefined }

    case 'specified':
      return { label: k.evtSpecified }

    case 'promoted':
      return { label: k.evtPromoted }

    case 'scheduled':
      return { label: k.evtScheduled, detail: str('reason') ?? undefined }

    case 'archived':
      return { label: k.evtArchived }

    case 'reprioritized':
      return { label: k.evtReprioritized(String(p.priority ?? '?')) }
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
  const k = useKanban()

  const act = (action: DiagnosticAction) => {
    if (action.kind === 'reclaim') {
      onReclaim()
    } else if (action.kind === 'cli_hint') {
      void navigator.clipboard.writeText(String(action.payload?.command ?? action.label))
      host.notify({ kind: 'info', message: k.commandCopied })
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
  const k = useKanban()
  const scope = useKanbanScope()
  const { data: roster } = useQuery({ queryKey: profilesKey(scope), queryFn: fetchProfiles, staleTime: 60_000 })

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
            <span className="text-(--ui-text-quaternary)">{k.unassigned}</span>
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
//
// On a RUNNING task the worker polls its comment thread and folds new notes
// into the live turn (OUT-OF-BAND steer), so a plain note reaches the agent
// mid-run within a few seconds — no block/unblock dance. `onRequeue` is the
// heavier option: post the note AND reclaim so the task restarts from scratch
// with the note in context (use when the current run has gone off the rails).
function CommentComposer({
  onRequeue,
  onSubmit,
  pending,
  running
}: {
  onRequeue?: (body: string) => void
  onSubmit: (body: string) => void
  pending: boolean
  running?: boolean
}) {
  const k = useKanban()
  const [body, setBody] = useState('')

  const submit = () => {
    const trimmed = body.trim()

    if (trimmed && !pending) {
      onSubmit(trimmed)
      setBody('')
    }
  }

  const requeue = () => {
    const trimmed = body.trim()

    if (trimmed && !pending && onRequeue) {
      onRequeue(trimmed)
      setBody('')
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative">
        <Textarea
          className={cn('field-sizing-content max-h-40 min-h-0 resize-none', running ? 'pr-[3.5rem]' : 'pr-[5rem]')}
          onChange={event => setBody(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={running ? k.messageWorker : k.addComment}
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
          {running ? k.send : k.comment}
        </Button>
      </div>
      {running && onRequeue && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.625rem] leading-tight text-(--ui-text-quaternary)">{k.deliveredLive}</span>
          <Button className="shrink-0" disabled={!body.trim() || pending} onClick={requeue} size="xs" variant="outline">
            <Codicon name="debug-restart" size="0.7rem" />
            {k.requeueWithNote}
          </Button>
        </div>
      )}
    </div>
  )
}

function DescriptionSection({ body, onSave }: { body: null | string | undefined; onSave: (body: string) => void }) {
  const k = useKanban()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  return (
    <Section
      action={
        <Button
          aria-label={editing ? k.cancelEdit : k.editDescription}
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
      label={k.description}
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
            {k.save}
          </Button>
        </div>
      ) : body ? (
        <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{body}</p>
      ) : (
        <p className="text-[0.8125rem] text-(--ui-text-quaternary)">{k.noDescription}</p>
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
  const k = useKanban()
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
            aria-label={k.uploadAttachment}
            disabled={pending}
            onClick={() => fileRef.current?.click()}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name={pending ? 'sync' : 'cloud-upload'} size="0.8rem" spinning={pending} />
          </Button>
        </>
      }
      label={k.attachments(attachments.length)}
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
        <p className="text-[0.75rem] text-(--ui-text-quaternary)">{k.noAttachments}</p>
      )}
    </Section>
  )
}

// Rough effort estimate via the auxiliary (auto-routed) model. Tokens +
// complexity, never dollars — providers don't report cost reliably. Gated
// behind an explicit click + disclaimer since it makes a model call. The
// control keeps a stable footprint (spinner swaps in place) so there's no
// layout jump when it runs.
function EstimateSection({ id }: { id: string }) {
  const k = useKanban()
  const [result, setResult] = useState<null | TaskEstimate>(null)

  const est = useMutation({
    mutationFn: () => estimateTask(id),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: r => {
      if (r.ok) {
        setResult(r)
      } else {
        host.notify({ kind: 'warning', message: r.reason || k.couldNotEstimate })
      }
    }
  })

  // A new task resets the cached estimate (the drawer reuses one instance).
  useEffect(() => setResult(null), [id])

  return (
    <Section label={k.estimate}>
      {result?.ok ? (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[0.8125rem]">
            <span className="font-medium tabular-nums text-(--ui-text-secondary)">
              ~{compactNumber(result.est_tokens)} {k.tokUnit}
            </span>
            {result.complexity && (
              <span className="text-(--ui-text-tertiary)">
                · {k.complexity[result.complexity] ?? result.complexity}
              </span>
            )}
            <Tip label={k.reEstimate}>
              <Button
                aria-label={k.reEstimate}
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
            {est.isPending ? k.estimating : k.estimateEffort}
          </Button>
          <Tip label={k.estimateTipLong}>
            <span className="text-[0.625rem] text-(--ui-text-quaternary)">{k.makesModelCall}</span>
          </Tip>
        </div>
      )}
    </Section>
  )
}

/** The slice of a board task the drawer's linked cards need. Kept structural so
 *  the board can hand its own resolver over without another round trip. */
export interface LinkedCard {
  assignee?: null | string
  id: string
  progress?: null | { done: number; total: number }
  status: string
  title: string
}

/**
 * One attempt's worth of activity: the events a single run emitted, or the
 * task-scoped rows that predate any run (``run_id`` null). Every attempt the
 * board still lists gets a row here, including one that emitted no events — an
 * attempt that crashed before its first event is exactly what an operator
 * opens the drawer to find.
 */
interface ActivityAttempt {
  events: KanbanEvent[]
  key: string
  run?: KanbanRun
}

/** Fold-row key for the run-less rows (a promotion, a link change, …). */
const TASK_SCOPED_ATTEMPT = 'task'

function attemptStart(attempt: ActivityAttempt): number {
  return attempt.run?.started_at ?? attempt.events[0]?.created_at ?? 0
}

/**
 * Group the feed into attempts. ``list_events`` returns created_at ASC, so the
 * last group is the newest attempt — the one left unfolded by default. Runs
 * with no events still get a row, so folding never hides an attempt entirely.
 */
function buildAttempts(events: KanbanEvent[], runs: KanbanRun[]): ActivityAttempt[] {
  const attempts = new Map<string, ActivityAttempt>()
  const runById = new Map(runs.map(run => [String(run.id), run]))

  for (const event of events) {
    const key = typeof event.run_id === 'number' ? String(event.run_id) : TASK_SCOPED_ATTEMPT
    const attempt = attempts.get(key) ?? { events: [], key, run: runById.get(key) }

    attempt.events.push(event)
    attempts.set(key, attempt)
  }

  for (const run of runs) {
    const key = String(run.id)

    if (!attempts.has(key)) {
      attempts.set(key, { events: [], key, run })
    }
  }

  return [...attempts.values()].sort((a, b) => attemptStart(a) - attemptStart(b))
}

/**
 * The activity feed, folded by attempt. A card retried six times used to render
 * every event and every run row at once — the height cap only hid them, it kept
 * none of them off the DOM. Folding each attempt behind one focusable row
 * leaves a single attempt on screen by default and still accounts for every
 * event on the card (folded attempts + unfolded events = events).
 */
function ActivityFeed({ events, runs, k }: { events: KanbanEvent[]; k: KanbanText; runs: KanbanRun[] }) {
  const attempts = buildAttempts(events, runs)
  // null = the operator has not touched the feed yet, so the newest attempt is
  // the unfolded one; deriving it keeps a late-landing detail query unfolded
  // instead of freezing the first (empty) render's choice.
  const [unfolded, setUnfolded] = useState<null | string[]>(null)
  const open = new Set(unfolded ?? (attempts.length > 0 ? [attempts[attempts.length - 1].key] : []))

  const toggle = (key: string) => {
    const next = new Set(open)

    if (next.has(key)) {
      next.delete(key)
    } else {
      next.add(key)
    }

    setUnfolded([...next])
  }

  return (
    <ul className="flex flex-col gap-1">
      {attempts.map((attempt, index) => {
        const run = attempt.run
        const events = attempt.events
        const foldable = events.length > 0
        const unfoldedHere = foldable && open.has(attempt.key)
        const failed = ['crashed', 'failed', 'timed_out', 'gave_up'].includes(run?.outcome ?? run?.status ?? '')
        const started = run?.started_at ?? events[0]?.created_at
        const note = run?.error ?? run?.summary
        const spent = duration(run?.started_at, run?.ended_at)

        return (
          <li className="flex flex-col gap-1" key={attempt.key}>
            <button
              aria-expanded={foldable ? unfoldedHere : undefined}
              aria-label={
                foldable ? (unfoldedHere ? k.collapse(`#${index + 1}`) : k.expand(`#${index + 1}`)) : undefined
              }
              className={cn(
                'flex items-center gap-2 rounded px-1 py-0.5 text-left text-[0.71rem]',
                foldable && 'hover:bg-(--chrome-action-hover)'
              )}
              data-attempt={attempt.key}
              data-events={events.length}
              disabled={!foldable}
              onClick={foldable ? () => toggle(attempt.key) : undefined}
              onKeyDown={event => {
                if (!foldable || !isSubmitEnter(event)) {
                  return
                }

                // Swallow Enter so the button's own activation cannot fire too.
                event.preventDefault()
                toggle(attempt.key)
              }}
              type="button"
            >
              {foldable ? (
                <Codicon
                  className="shrink-0 text-(--ui-text-tertiary)"
                  name={unfoldedHere ? 'chevron-down' : 'chevron-right'}
                  size="0.7rem"
                />
              ) : (
                <span className="w-[0.7rem] shrink-0" />
              )}
              <span className="shrink-0 tabular-nums text-(--ui-text-quaternary)">{index + 1}</span>
              {run && (
                <Badge size="xs" variant={failed ? 'destructive' : 'muted'}>
                  {run.outcome ?? run.status}
                </Badge>
              )}
              {run?.profile && <span className="shrink-0 text-(--ui-text-tertiary)">{run.profile}</span>}
              {spent && <span className="shrink-0 text-(--ui-text-quaternary)">{spent}</span>}
              {started && <span className="shrink-0 text-(--ui-text-quaternary)">{ago(started)}</span>}
              <span className="ml-auto shrink-0 tabular-nums text-(--ui-text-quaternary)">{events.length}</span>
            </button>
            {note && (
              <p
                className={cn(
                  'line-clamp-1 px-1 text-[0.6875rem]',
                  run?.error ? 'text-destructive' : 'text-(--ui-text-quaternary)'
                )}
              >
                {note}
              </p>
            )}
            {unfoldedHere && (
              <ScrollFade deps={events.length} max="7rem">
                <ul className="flex flex-col gap-1 pl-4">
                  {events.map(event => {
                    const { detail: extra, label } = eventText(event, k)

                    return (
                      <li
                        className="flex items-baseline gap-2 text-[0.6875rem]"
                        data-event-id={event.id}
                        key={event.id}
                      >
                        <span className="shrink-0 text-(--ui-text-secondary)">{label}</span>
                        {extra && (
                          <span className="min-w-0 truncate text-[0.625rem] text-(--ui-text-quaternary)" title={extra}>
                            {extra}
                          </span>
                        )}
                        <span className="ml-auto shrink-0 text-(--ui-text-quaternary)">{ago(event.created_at)}</span>
                      </li>
                    )
                  })}
                </ul>
              </ScrollFade>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export function TaskDrawer({
  columns,
  focusLinks,
  id,
  lookup = () => undefined,
  onClose,
  onOpen
}: {
  columns: string[]
  focusLinks?: number
  id: null | string
  /** Board-wide id → task resolver: the parent/child cards below read their
   *  status, assignee and progress off the board instead of another fetch. */
  lookup?: (id: string) => undefined | LinkedCard
  onClose: () => void
  onOpen: (id: string) => void
}) {
  const k = useKanban()
  const qc = useQueryClient()
  const scope = useKanbanScope()
  const slug = useValue($boardSlug)
  const linksRef = useRef<HTMLDivElement>(null)

  // Socket-invalidated (bindApi); the interval is only the socketless heartbeat.
  const { data: detail, error } = useQuery({
    enabled: query => !!id && routedToScope(query),
    queryFn: () => fetchTask(id!),
    queryKey: taskKey(scope, slug, id ?? ''),
    refetchInterval: 30_000
  })

  const task = detail?.task
  const running = task?.status === 'running'
  const defaultAssignee = useDefaultAssignee()

  const { data: log } = useQuery({
    enabled: query => !!id && routedToScope(query),
    queryFn: () => fetchLog(id!),
    queryKey: logKey(scope, slug, id ?? ''),
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

  // The card menu's "add link / add child" bumps `focusLinks`; scroll the
  // dependencies section into view once per bump (and only after the fetch has
  // rendered something to scroll to).
  const [scrolledPing, setScrolledPing] = useState<null | number>(null)

  useEffect(() => {
    if (!focusLinks || focusLinks === scrolledPing || !detail) {
      return
    }

    setScrolledPing(focusLinks)
    linksRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [detail, focusLinks, scrolledPing])

  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: taskKey(scope, slug, id!) })
    void qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
  }

  // Optimistic status change against the task cache; rolls back + toasts on a
  // rejected transition (the backend enforces the workflow).
  const moveMut = useMutation({
    mutationFn: (status: string) => patchTask(id!, { status }),
    onMutate: async status => {
      await qc.cancelQueries({ queryKey: taskKey(scope, slug, id!) })
      const previous = qc.getQueryData<KanbanTaskDetail>(taskKey(scope, slug, id!))

      if (previous) {
        qc.setQueryData(taskKey(scope, slug, id!), { ...previous, task: { ...previous.task, status } })
      }

      return { previous }
    },
    onError: (err, _status, context) => {
      if (context?.previous) {
        qc.setQueryData(taskKey(scope, slug, id!), context.previous)
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

  // "Note & requeue" for a running task: post the note, then reclaim so the
  // dispatcher re-runs it with the note in the worker's context — the one-click
  // replacement for the block → comment → unblock dance.
  const requeueMut = useMutation({
    mutationFn: async (body: string) => {
      await addComment(id!, body)
      await reclaimTask(id!)
    },
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      host.notify({ kind: 'info', message: k.notePosted })
      invalidate()
    }
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
      host.notify({ kind: 'info', message: lockedReason(k, status) })

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
                    aria-label={k.taskActions}
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
                      host.notify({ kind: 'info', message: k.copiedId(task.id) })
                    }}
                  >
                    <Codicon name="copy" size="0.85rem" />
                    {k.copyTaskId}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => {
                      void navigator.clipboard.writeText(task.title || task.id)
                      host.notify({ kind: 'info', message: k.copiedTitle })
                    }}
                  >
                    <Codicon name="copy" size="0.85rem" />
                    {k.copyTitle}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={mutate(() => patchTask(task.id, { status: 'archived' }), onClose)}>
                    <Codicon name="archive" size="0.85rem" />
                    {k.archive}
                  </DropdownMenuItem>
                  <DropdownMenuItem className="text-destructive" onSelect={mutate(() => deleteTask(task.id), onClose)}>
                    <Codicon name="trash" size="0.85rem" />
                    {k.delete}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            <button
              aria-label={k.close}
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
              <MetaRow label={k.assignee}>
                <AssigneeMenu
                  current={task.assignee}
                  onReassign={profile => void mutate(() => reassignTask(task.id, profile))()}
                />
              </MetaRow>
              {typeof task.priority === 'number' && <MetaRow label={k.metaPriority}>{task.priority}</MetaRow>}
              {task.tenant && <MetaRow label={k.metaTenant}>{task.tenant}</MetaRow>}
              {task.workspace_path && (
                <MetaRow label={k.workspace}>
                  {task.workspace_kind ? `${task.workspace_kind}: ` : ''}
                  {task.workspace_path}
                </MetaRow>
              )}
              <MetaRow label={k.model}>
                <ModelOverrideField
                  onChange={next => void mutate(() => patchTask(task.id, overridePatch(next)))()}
                  value={{
                    effort: task.reasoning_effort ?? '',
                    model: task.model_override ?? '',
                    provider: task.provider_override ?? ''
                  }}
                />
              </MetaRow>
              {task.created_by && <MetaRow label={k.metaCreatedBy}>{task.created_by}</MetaRow>}
              {ago(task.created_at) && <MetaRow label={k.metaCreated}>{ago(task.created_at)}</MetaRow>}
              {running && task.worker_pid ? <MetaRow label={k.metaWorkerPid}>{task.worker_pid}</MetaRow> : null}
            </div>

            {task.status === 'ready' && !task.assignee && !defaultAssignee && (
              <Callout title={k.readyUnassignedTitle} tone={SEVERITY_TONE.warning}>
                <p className="text-[0.71rem] leading-relaxed text-(--ui-text-secondary)">{k.readyUnassignedBody}</p>
              </Callout>
            )}

            {task.diagnostics && task.diagnostics.length > 0 && (
              <Section label={k.diagnosticsN(task.diagnostics.length)}>
                <Diagnostics items={task.diagnostics} onReclaim={() => void mutate(() => reclaimTask(task.id))()} />
              </Section>
            )}

            <DescriptionSection body={task.body} onSave={body => void mutate(() => patchTask(task.id, { body }))()} />

            <EstimateSection id={task.id} />

            {task.result && (
              <Section label={k.result}>
                <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{task.result}</p>
              </Section>
            )}

            {task.latest_summary && !isAdminSummary(task.latest_summary) && (
              <Section label={k.latestSummary}>
                <p className="whitespace-pre-wrap text-[0.8125rem] text-(--ui-text-secondary)">{task.latest_summary}</p>
              </Section>
            )}

            {(detail.links.parents.length > 0 || detail.links.children.length > 0 || !!focusLinks) && (
              <Section label={k.dependencies}>
                <div className="flex flex-col gap-1.5" ref={linksRef}>
                  {(['parents', 'children'] as const).map(side =>
                    detail.links[side].length > 0 ? (
                      <div className="flex flex-col gap-1" key={side}>
                        <span className="text-[0.6875rem] text-(--ui-text-quaternary)">
                          {side === 'parents' ? k.blockedBy : k.blocks}
                        </span>
                        {detail.links[side].map(linked => {
                          const card = lookup(linked)

                          // Off-board relatives (filtered out, archived elsewhere)
                          // keep the old id chip rather than inventing a card.
                          if (!card) {
                            return (
                              <button
                                className="w-fit rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5 font-mono text-[0.625rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
                                key={linked}
                                onClick={() => onOpen(linked)}
                                type="button"
                              >
                                {shortId(linked)}
                              </button>
                            )
                          }

                          return (
                            <button
                              className="flex items-center gap-1.5 rounded border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-1.5 py-1 text-left transition-colors hover:bg-(--chrome-action-hover)"
                              key={linked}
                              onClick={() => onOpen(linked)}
                              type="button"
                            >
                              <span
                                aria-hidden
                                className="size-1.5 shrink-0 rounded-full"
                                style={{ backgroundColor: DEP_TONES[depSegment(card.status)] }}
                              />
                              <span className="min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-secondary)">
                                {card.title || card.id}
                              </span>
                              {card.assignee && (
                                <span className="shrink-0 font-mono text-[0.625rem] text-(--ui-text-quaternary)">
                                  {card.assignee}
                                </span>
                              )}
                              {card.progress && card.progress.total > 0 && (
                                <span className="flex shrink-0 items-center gap-1">
                                  <span className="h-[3px] w-8 overflow-hidden rounded-full bg-(--ui-bg-quaternary)">
                                    <span
                                      className="block h-full rounded-full bg-(--ui-text-secondary)"
                                      style={{
                                        width: `${Math.round((card.progress.done / card.progress.total) * 100)}%`
                                      }}
                                    />
                                  </span>
                                  <span className="font-mono text-[0.625rem] text-(--ui-text-quaternary)">
                                    {card.progress.done}/{card.progress.total}
                                  </span>
                                </span>
                              )}
                            </button>
                          )
                        })}
                      </div>
                    ) : focusLinks ? (
                      <div className="flex items-center gap-1.5" key={side}>
                        <span className="text-[0.6875rem] text-(--ui-text-quaternary)">
                          {side === 'parents' ? k.blockedBy : k.blocks}
                        </span>
                        <span className="text-[0.75rem] text-(--ui-text-quaternary)">—</span>
                      </div>
                    ) : null
                  )}
                </div>
              </Section>
            )}

            <Section
              action={
                <Tip label={running ? k.commentsHelpRunning : k.commentsHelp}>
                  <span className="grid size-5 place-items-center rounded text-(--ui-text-quaternary) hover:text-(--ui-text-secondary)">
                    <Codicon name="question" size="0.8rem" />
                  </span>
                </Tip>
              }
              label={k.comments(detail.comments.length)}
            >
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
              <CommentComposer
                onRequeue={body => requeueMut.mutate(body)}
                onSubmit={body => commentMut.mutate(body)}
                pending={commentMut.isPending || requeueMut.isPending}
                running={running}
              />
            </Section>

            {(detail.events.length > 0 || detail.runs.length > 0) && (
              <Section
                action={
                  detail.runs.length > 0 ? <span className={FIELD_LABEL}>{k.runs(detail.runs.length)}</span> : undefined
                }
                label={k.activity(detail.events.length)}
              >
                <ActivityFeed events={detail.events} k={k} runs={detail.runs} />
              </Section>
            )}

            {log?.exists && log.content && (
              <Section label={log.truncated ? k.workerLogTail : k.workerLog}>
                <ScrollFade deps={log.content.length} max="12rem">
                  <LogView className="border-0 px-0">{log.content}</LogView>
                </ScrollFade>
              </Section>
            )}

            {Array.isArray(detail.attachments) && (
              <AttachmentsSection
                attachments={detail.attachments}
                onUpload={file => uploadMut.mutate(file)}
                pending={uploadMut.isPending}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
