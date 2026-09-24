/**
 * Task modal — the desktop port of the dashboard's task detail, Linear-style:
 * a centered two-column dialog (main: diagnostics, description, result, and the
 * feed folded by attempt; right sidebar: property rows with the inline editors,
 * dependency chips carrying their gated-state tone), on the shared Dialog.
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  compactNumber,
  CopyButton,
  Dialog,
  DialogContent,
  DialogTitle,
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
  MessageTextContent,
  SegmentedControl,
  Textarea,
  Tip,
  useI18n,
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
  type TaskEstimate,
  type WorkerLog
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
  isLockedTarget,
  type KanbanText,
  lockedReason,
  PriorityGlyph,
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

// Task bodies, results, summaries and comments are agent-written markdown;
// rendering them raw left `**Goal:**` and backticks literal. Same renderer as
// chat. `media={false}`: kanban text is not session-scoped, so `MEDIA:` paths
// must not be resolved against the active gateway.
function TaskMarkdown({ text }: { text: string }) {
  return (
    <div className="min-w-0 [&_.aui-md>:first-child]:mt-0 [&_.aui-md>:last-child]:mb-0" data-selectable-text="true">
      <MessageTextContent media={false} text={text} />
    </div>
  )
}

// Sidebar property row: a Section (so every sidebar label — these, Estimate,
// Attachments — shares FIELD_LABEL and one rhythm) whose value slot holds the
// inline editors. Values wrap anywhere so a long path never clips at the edge.
function MetaRow({ children, label }: { children: ReactNode; label: string }) {
  return (
    <Section label={label}>
      <div className="min-w-0 text-[0.75rem] text-(--ui-text-secondary) [overflow-wrap:anywhere]">{children}</div>
    </Section>
  )
}

// The task's workspace: the kind as a badge when it says more than "a
// directory", the path in mono (wrapping), and a copy affordance.
function WorkspaceValue({ kind, path }: { kind: null | string | undefined; path: string }) {
  return (
    <div className="flex items-start gap-1.5">
      <div className="flex min-w-0 flex-1 flex-col items-start gap-1">
        {kind && kind !== 'dir' && (
          <Badge size="xs" variant="muted">
            {kind}
          </Badge>
        )}
        <span className="font-mono text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{path}</span>
      </div>
      <CopyButton
        appearance="icon"
        buttonSize="icon-xs"
        buttonVariant="ghost"
        className="-mt-0.5 shrink-0"
        text={path}
      />
    </div>
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
            <p className="whitespace-pre-wrap text-[0.6875rem] leading-relaxed text-(--ui-text-secondary)">
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

  const empty = !body.trim() || pending
  const sendLabel = running ? k.send : k.comment

  // Sized like the new-project idea field (default control padding, inset icon
  // action); a labelled text button floating inside the textarea read as part
  // of the input.
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative">
        <Textarea
          className="field-sizing-content max-h-40 resize-none pr-9 text-[0.8125rem]"
          onChange={event => setBody(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={running ? k.messageWorker : k.addComment}
          value={body}
        />
        <Tip label={sendLabel}>
          <Button
            aria-label={sendLabel}
            className="absolute top-1 right-1 text-muted-foreground/80 hover:text-foreground"
            disabled={empty}
            onClick={submit}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="arrow-up" size="0.85rem" />
          </Button>
        </Tip>
      </div>
      {running && onRequeue && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.625rem] leading-tight text-(--ui-text-quaternary)">{k.deliveredLive}</span>
          <Button className="shrink-0" disabled={empty} onClick={requeue} size="xs" variant="outline">
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
        <TaskMarkdown text={body} />
      ) : (
        <p className="text-[0.8125rem] text-(--ui-text-quaternary)">{k.noDescription}</p>
      )}
    </Section>
  )
}

// `latest_summary` is just the newest non-null run summary. A reclaim writes an
// administrative note into that slot; hide those (Runs still shows them).
const isAdminSummary = (summary: string) => /^status changed to \w+ \(dashboard\/direct\)$/.test(summary)

// The filename is the download action. The path is the backend's own
// stored_path, saved through the connection/profile that returned this detail;
// a row without one (older backend) stays inert rather than guessing a path.
function AttachmentDownload({
  attachment,
  onDownload
}: {
  attachment: KanbanAttachment
  onDownload: (path: string, suggestedName: string) => Promise<void>
}) {
  const { t } = useI18n()
  const path = attachment.stored_path?.trim()

  const download = useMutation({
    mutationFn: () => onDownload(path!, attachment.filename)
  })

  // Long names truncate in the narrow sidebar; the tip reveals the full name.
  return (
    <Tip label={attachment.filename} placement="row">
      <Button
        aria-label={`${t.fileMenu.download} ${attachment.filename}`}
        className="max-w-full justify-start font-normal"
        disabled={!path || download.isPending}
        onClick={() => download.mutate()}
        size="inline"
        variant="text"
      >
        <Codicon name={download.isPending ? 'sync' : 'cloud-download'} size="0.75rem" spinning={download.isPending} />
        <span className="truncate">{attachment.filename}</span>
      </Button>
    </Tip>
  )
}

function AttachmentsSection({
  attachments,
  onDownload,
  onUpload,
  pending
}: {
  attachments: KanbanAttachment[]
  onDownload: (path: string, suggestedName: string) => Promise<void>
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
              <AttachmentDownload attachment={attachment} onDownload={onDownload} />
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

// Sidebar dependency chips: one wrap of title-labeled buttons per side
// (Blocked by = parents, Blocks = children). Titles come from the backend's
// `link_tasks`; ids remain in the tooltip + as the fallback label.
function LinkChips({
  ids,
  linkTitles,
  onOpen,
  statusOf
}: {
  ids: string[]
  linkTitles: Map<string, string>
  onOpen: (id: string) => void
  statusOf?: (id: string) => string | undefined
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {ids.map(linked => {
        const label = linkTitles.get(linked) || shortId(linked)

        // Chips truncate in the narrow sidebar; the tip reveals the full title.
        return (
          <Tip key={linked} label={label} placement="row">
            <button
              aria-label={label}
              className="max-w-full truncate rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5 text-[0.6875rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
              onClick={() => onOpen(linked)}
              type="button"
            >
              <span
                aria-hidden
                className="size-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: DEP_TONES[depSegment(statusOf?.(linked))] }}
              />
              {label}
            </button>
          </Tip>
        )
      })}
    </div>
  )
}

/** The main column's feed: Comments (default) / Activity / Runs / Worker log,
 *  selected Jira-style with a segmented control ("Activity ▸ Show: …"). The
 *  control hides itself when only comments exist — nothing to switch to. */
function FeedTabs({
  commentPending,
  detail,
  log,
  onComment,
  onRequeue,
  running
}: {
  commentPending: boolean
  detail: KanbanTaskDetail
  log: null | WorkerLog
  onComment: (body: string) => void
  onRequeue: (body: string) => void
  running: boolean
}) {
  const k = useKanban()
  const [tab, setTab] = useState<'activity' | 'comments' | 'log' | 'runs'>('comments')

  const hasLog = !!log?.exists && !!log.content
  const switchable = detail.events.length > 0 || detail.runs.length > 0 || hasLog

  const tabs = [
    { id: 'comments' as const, label: k.comments(detail.comments.length) },
    { id: 'activity' as const, label: k.activity(detail.events.length) },
    { id: 'runs' as const, label: k.runs(detail.runs.length) },
    { id: 'log' as const, label: k.workerLog }
  ].filter(
    t =>
      t.id === 'comments' ||
      (t.id === 'activity'
        ? detail.events.length > 0 || detail.runs.length > 0
        : t.id === 'runs'
          ? detail.runs.length > 0
          : hasLog)
  )

  const help = (
    <Tip label={running ? k.commentsHelpRunning : k.commentsHelp}>
      <span className="grid size-5 place-items-center rounded text-(--ui-text-quaternary) hover:text-(--ui-text-secondary)">
        <Codicon name="question" size="0.8rem" />
      </span>
    </Tip>
  )

  const body = (
    <div className="flex flex-col gap-4">
      {tab === 'comments' && (
        <>
          {detail.comments.length > 0 && (
            <ul className="flex flex-col gap-3">
              {detail.comments.map(comment => (
                <li className="flex flex-col gap-0.5" key={comment.id}>
                  <div className="flex items-baseline gap-2 text-[0.75rem]">
                    <span className="font-medium text-(--ui-text-secondary)">{comment.author}</span>
                    <span className="text-[0.625rem] text-(--ui-text-quaternary)">{ago(comment.created_at)}</span>
                  </div>
                  <TaskMarkdown text={comment.body} />
                </li>
              ))}
            </ul>
          )}
          <CommentComposer onRequeue={onRequeue} onSubmit={onComment} pending={commentPending} running={running} />
        </>
      )}
      {tab === 'activity' && <ActivityFeed events={detail.events} k={k} runs={detail.runs} />}
      {tab === 'runs' && (
        <ScrollFade max="11rem">
          <ul className="flex flex-col gap-1.5">
            {detail.runs.map(run => {
              const failed = ['crashed', 'failed', 'timed_out', 'gave_up'].includes(run.outcome ?? run.status)

              return (
                <li className="flex flex-col gap-0.5 text-[0.6875rem]" key={run.id}>
                  <div className="flex items-center gap-2">
                    <Badge size="xs" variant={failed ? 'destructive' : 'muted'}>
                      {run.outcome ?? run.status}
                    </Badge>
                    {run.profile && <span className="text-(--ui-text-tertiary)">{run.profile}</span>}
                    {duration(run.started_at, run.ended_at) && (
                      <span className="text-(--ui-text-quaternary)">{duration(run.started_at, run.ended_at)}</span>
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
      )}
      {tab === 'log' && (
        <ScrollFade deps={log?.content.length} max="12rem">
          <LogView className="border-0 px-0">{log!.content}</LogView>
        </ScrollFade>
      )}
    </div>
  )

  // With something to switch to, the segmented control IS the heading — a
  // label above it only repeated the active tab ("Comments · 2" twice).
  if (!switchable) {
    return (
      <Section action={help} label={k.comments(detail.comments.length)}>
        {body}
      </Section>
    )
  }

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <SegmentedControl onChange={setTab} options={tabs} value={tab} />
        {help}
      </div>
      {body}
    </section>  )
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

  // Linked tasks resolved to titles by the backend (`link_tasks`); absent on
  // older backends, where the chips fall back to short ids.
  const linkTitles = new Map((detail?.link_tasks ?? []).map(linked => [linked.id, linked.title]))
  const linkStatus = new Map((detail?.link_tasks ?? []).map(linked => [linked.id, linked.status]))

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

  // The shared Dialog owns the chrome tokens, focus trap, Esc and outside-click
  // dismissal, and publishes itself as the portal container so the status,
  // assignee, actions and model menus open inside it (no z-index rung needed).
  // The body box is split into two independently scrolling columns, so it
  // clips instead of scrolling itself; its height follows the content up to
  // the cap.
  return (
    <Dialog onOpenChange={open => !open && onClose()} open>
      <DialogContent
        aria-describedby={undefined}
        bodyClassName="flex max-h-[min(84vh,54rem)] flex-col gap-0 overflow-hidden p-0"
        className="w-[min(62rem,94vw)] max-w-none"
        showCloseButton={false}
      >
        <header className="flex flex-col gap-2 px-5 pt-4 pb-3">
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
                    <Button aria-label={k.taskActions} size="icon-xs" variant="ghost">
                      <Codicon name="ellipsis" />
                    </Button>
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
                    <DropdownMenuItem
                      className="text-destructive"
                      onSelect={mutate(() => deleteTask(task.id), onClose)}
                    >
                      <Codicon name="trash" size="0.85rem" />
                      {k.delete}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
              <Button aria-label={k.close} onClick={onClose} size="icon-xs" variant="ghost">
                <Codicon name="close" />
              </Button>
            </div>
          </div>
          <DialogTitle className="leading-snug" data-selectable-text="true">
            {task ? task.title || task.id : shortId(id)}
          </DialogTitle>
        </header>

        <div className="flex min-h-0 flex-1 flex-col" data-selectable-text="true">
          {errorMessage ? (
            <ErrorState title={errorMessage} />
          ) : !detail || !task ? (
            <div className="grid h-32 place-items-center">
              <Loader type="lemniscate-bloom" />
            </div>
          ) : (
            <div className="flex min-h-0 flex-1">
              <div className="min-w-0 flex-1 overflow-y-auto px-5 pb-5">
                <div className="flex flex-col gap-5">
                  {task.status === 'ready' && !task.assignee && !defaultAssignee && (
                    <Callout title={k.readyUnassignedTitle} tone={SEVERITY_TONE.warning}>
                      <p className="text-[0.6875rem] leading-relaxed text-(--ui-text-secondary)">
                        {k.readyUnassignedBody}
                      </p>
                    </Callout>
                  )}

                  {task.diagnostics && task.diagnostics.length > 0 && (
                    <Section label={k.diagnosticsN(task.diagnostics.length)}>
                      <Diagnostics
                        items={task.diagnostics}
                        onReclaim={() => void mutate(() => reclaimTask(task.id))()}
                      />
                    </Section>
                  )}

                  <DescriptionSection
                    body={task.body}
                    onSave={body => void mutate(() => patchTask(task.id, { body }))()}
                  />

                  {task.result && (
                    <Section label={k.result}>
                      <TaskMarkdown text={task.result} />
                    </Section>
                  )}

                  {task.latest_summary && !isAdminSummary(task.latest_summary) && (
                    <Section label={k.latestSummary}>
                      <TaskMarkdown text={task.latest_summary} />
                    </Section>
                  )}

                  <FeedTabs
                    commentPending={commentMut.isPending || requeueMut.isPending}
                    detail={detail}
                    log={log ?? null}
                    onComment={body => commentMut.mutate(body)}
                    onRequeue={body => requeueMut.mutate(body)}
                    running={running}
                  />
                </div>
              </div>
              <aside className="flex w-64 shrink-0 flex-col gap-4 overflow-y-auto border-l border-(--ui-stroke-tertiary) px-4 pb-5">
                <MetaRow label={k.assignee}>
                  <AssigneeMenu
                    current={task.assignee}
                    onReassign={profile => void mutate(() => reassignTask(task.id, profile))()}
                  />
                </MetaRow>
                {typeof task.priority === 'number' && (
                  <MetaRow label={k.metaPriority}>
                    <PriorityGlyph priority={task.priority} />
                  </MetaRow>
                )}
                {task.tenant && <MetaRow label={k.metaTenant}>{task.tenant}</MetaRow>}
                {task.workspace_path && (
                  <MetaRow label={k.workspace}>
                    <WorkspaceValue kind={task.workspace_kind} path={task.workspace_path} />
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
                {(detail.links.parents.length > 0 || detail.links.children.length > 0 || !!focusLinks) && (
                  <div className="flex flex-col gap-4" ref={linksRef}>
                    {(['parents', 'children'] as const).map(side =>
                      detail.links[side].length > 0 ? (
                        <MetaRow key={side} label={side === 'parents' ? k.blockedBy : k.blocks}>
                          <LinkChips
                            ids={detail.links[side]}
                            linkTitles={linkTitles}
                            onOpen={onOpen}
                            statusOf={id => lookup(id)?.status ?? linkStatus.get(id)}
                          />
                        </MetaRow>
                      ) : focusLinks ? (
                        <MetaRow key={side} label={side === 'parents' ? k.blockedBy : k.blocks}>
                          <span className="text-[0.75rem] text-(--ui-text-quaternary)">—</span>
                        </MetaRow>
                      ) : null
                    )}
                  </div>
                )}
                {task.created_by && <MetaRow label={k.metaCreatedBy}>{task.created_by}</MetaRow>}
                {ago(task.created_at) && <MetaRow label={k.metaCreated}>{ago(task.created_at)}</MetaRow>}
                {running && task.worker_pid ? <MetaRow label={k.metaWorkerPid}>{task.worker_pid}</MetaRow> : null}
                <EstimateSection id={task.id} />

                {Array.isArray(detail.attachments) && (
                  <AttachmentsSection
                    attachments={detail.attachments}
                    onDownload={detail.downloadAttachment}
                    onUpload={file => uploadMut.mutate(file)}
                    pending={uploadMut.isPending}
                  />
                )}
              </aside>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
