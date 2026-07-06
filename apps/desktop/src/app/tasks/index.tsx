import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { TextTab } from '@/components/ui/text-tab'
import { Textarea } from '@/components/ui/textarea'
import {
  createCronJob,
  deleteCronJob,
  getCronJobRuns,
  getSessionMessages,
  triggerCronJob,
  type CronJob,
  type SessionInfo,
  type SessionMessage
} from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle, Clock, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { updateCronJobs } from '@/store/cron'
import { notify, notifyError } from '@/store/notifications'
import { $tasks } from '@/store/tasks'

import { jobTitle } from '../cron/job-state'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayMain, OverlayNewButton, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import {
  deriveProgress,
  isStuck,
  primaryRun,
  scheduleStringForWhen,
  taskPhase,
  type TaskPhase,
  type TaskProgress,
  type TaskWhen
} from './task-model'

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')
const truncate = (value: string, max = 60): string => (value.length > max ? `${value.slice(0, max)}…` : value)

function jobPrompt(job: CronJob): string {
  return asText(job.prompt)
}

function formatTime(iso?: null | string): string {
  if (!iso) {
    return '—'
  }

  const date = new Date(iso)

  return Number.isNaN(date.valueOf()) ? iso : date.toLocaleString()
}

function formatEpoch(seconds?: null | number): string {
  if (!seconds) {
    return '—'
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.valueOf()) ? '—' : date.toLocaleString()
}

type Tab = 'running' | 'done'

interface TasksViewProps extends React.ComponentProps<'section'> {
  onOpenSession?: (sessionId: string) => void
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function TasksView({ onOpenSession, setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: TasksViewProps) {
  const { t } = useI18n()
  const c = t.tasks
  // Projection over the shared cron atom (controller already polls it), so the
  // list stays live without a second fetch. Only the first paint waits.
  const tasks = useStore($tasks)
  const [loading, setLoading] = useState(tasks.length === 0)
  const [tab, setTab] = useState<Tab>('running')
  const [selectedId, setSelectedId] = useState<null | string>(null)
  const [busyId, setBusyId] = useState<null | string>(null)
  const [creating, setCreating] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<CronJob | null>(null)
  const [deleting, setDeleting] = useState(false)
  // Re-render the stuck check on a slow timer so "no activity for 20m" flips
  // without a data change.
  const [nowTick, setNowTick] = useState(() => Date.now())

  // The tasks list is driven by the controller's cron poll; a manual refresh
  // just re-triggers it via the same hotkey the cron page uses (best-effort).
  const refresh = useCallback(async () => {
    // Nothing to fetch here directly — the controller owns the cron poll. Clear
    // the first-paint gate in case the atom was already warm.
    setLoading(false)
  }, [])

  useRefreshHotkey(refresh)

  useEffect(() => {
    if (tasks.length > 0) {
      setLoading(false)
    }
  }, [tasks.length])

  useEffect(() => {
    const id = window.setInterval(() => setNowTick(Date.now()), 60_000)

    return () => window.clearInterval(id)
  }, [])

  const running = useMemo(() => tasks.filter(task => taskPhase(task) === 'running'), [tasks])
  const finished = useMemo(() => tasks.filter(task => taskPhase(task) !== 'running'), [tasks])
  const visible = tab === 'running' ? running : finished

  // Detail always reflects a concrete task in the active tab.
  const selected = useMemo(
    () => visible.find(task => task.id === selectedId) ?? visible[0] ?? null,
    [visible, selectedId]
  )

  async function handleCreate(prompt: string, when: TaskWhen) {
    setCreating(true)

    try {
      const created = await createCronJob({
        prompt,
        schedule: scheduleStringForWhen(when),
        deliver: 'local'
      })

      updateCronJobs(rows => [...rows, created])
      notify({ kind: 'success', title: c.created, message: truncate(jobTitle(created)) })
      setTab('running')
      setSelectedId(created.id)
    } finally {
      setCreating(false)
    }
  }

  async function handleRunNow(job: CronJob) {
    setBusyId(job.id)

    try {
      const updated = await triggerCronJob(job.id)
      updateCronJobs(rows => rows.map(row => (row.id === job.id ? updated : row)))
      notify({ kind: 'success', title: c.startedNow, message: truncate(jobTitle(job)) })
    } catch (err) {
      notifyError(err, c.failedStart)
    } finally {
      setBusyId(null)
    }
  }

  async function handleConfirmDelete() {
    if (!pendingDelete) {
      return
    }

    setDeleting(true)

    try {
      await deleteCronJob(pendingDelete.id)
      updateCronJobs(rows => rows.filter(row => row.id !== pendingDelete.id))
      notify({ kind: 'success', title: c.deleted, message: truncate(jobTitle(pendingDelete)) })
      setPendingDelete(null)
    } catch (err) {
      notifyError(err, c.failedDelete)
    } finally {
      setDeleting(false)
    }
  }

  const [editorOpen, setEditorOpen] = useState(false)

  return (
    <section
      {...props}
      className="flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)"
    >
      {loading && tasks.length === 0 ? (
        <PageLoader label={c.loading} />
      ) : (
        <OverlaySplitLayout>
          <OverlaySidebar>
            <OverlayNewButton icon="rocket" label={c.newTask} onClick={() => setEditorOpen(true)} />
            <div className="mb-1 flex items-center gap-3 px-2">
              <TextTab active={tab === 'running'} onClick={() => setTab('running')}>
                {c.tabRunning}
                {running.length > 0 ? ` · ${running.length}` : ''}
              </TextTab>
              <TextTab active={tab === 'done'} onClick={() => setTab('done')}>
                {c.tabDone}
                {finished.length > 0 ? ` · ${finished.length}` : ''}
              </TextTab>
            </div>
            {visible.map(task => (
              <TaskListRow
                active={selected?.id === task.id}
                job={task}
                key={task.id}
                now={nowTick}
                onSelect={() => setSelectedId(task.id)}
              />
            ))}
            {visible.length === 0 && (
              <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                {tab === 'running' ? c.emptyRunning : c.emptyDone}
              </p>
            )}
          </OverlaySidebar>

          <OverlayMain className="px-0">
            {selected ? (
              <TaskDetail
                busy={busyId === selected.id}
                c={c}
                job={selected}
                now={nowTick}
                onDelete={() => setPendingDelete(selected)}
                onOpenSession={onOpenSession}
                onRunNow={() => void handleRunNow(selected)}
              />
            ) : (
              <div className="grid h-full place-items-center px-6 py-12 text-center text-sm text-muted-foreground">
                <div>
                  <Sparkles className="mx-auto size-6 text-muted-foreground/60" />
                  <p className="mt-3 max-w-sm">{c.emptyDetail}</p>
                  <Button className="mt-4" onClick={() => setEditorOpen(true)} size="sm" variant="outline">
                    <Codicon name="rocket" size="0.875rem" />
                    {c.newTask}
                  </Button>
                </div>
              </div>
            )}
          </OverlayMain>
        </OverlaySplitLayout>
      )}

      <TaskEditorDialog
        c={c}
        commonCopy={t.common}
        onClose={() => setEditorOpen(false)}
        onCreate={handleCreate}
        open={editorOpen}
        submitting={creating}
      />

      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{c.deleteTitle}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? (
                <>
                  {c.deleteDescPrefix}
                  <span className="font-medium text-foreground">{truncate(jobTitle(pendingDelete))}</span>
                  {c.deleteDescSuffix}
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={deleting} onClick={() => void handleConfirmDelete()} variant="destructive">
              {deleting ? c.deleting : t.common.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}

// ── List row (status pip + title + one-line progress summary) ────────────────

const PHASE_DOT: Record<TaskPhase, string> = {
  running: 'bg-primary',
  done: 'bg-(--ui-text-quaternary)',
  failed: 'bg-destructive'
}

function TaskListRow({
  active,
  job,
  now,
  onSelect
}: {
  active: boolean
  job: CronJob
  now: number
  onSelect: () => void
}) {
  const phase = taskPhase(job)
  const running = phase === 'running'

  return (
    <button
      className={cn(
        'flex w-full items-center gap-2.5 rounded-[0.625rem] px-2 py-2 text-left transition-colors',
        active ? 'bg-accent text-foreground' : 'text-foreground/85 hover:bg-accent/60'
      )}
      data-task-row={job.id}
      onClick={onSelect}
      type="button"
    >
      <span
        aria-hidden="true"
        className={cn(
          'size-3 shrink-0 rounded-full',
          PHASE_DOT[phase],
          running && 'motion-safe:animate-pulse'
        )}
      />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="min-w-0 truncate text-sm font-medium leading-tight">{jobTitle(job)}</span>
        <TaskRowSubtitle job={job} now={now} />
      </span>
    </button>
  )
}

// Row subtitle: schedule/when text, or a "stuck" flag when a running task has
// gone quiet. Kept cheap (no run fetch) — the detail pane does the deep read.
function TaskRowSubtitle({ job, now }: { job: CronJob; now: number }) {
  const { t } = useI18n()
  const display = asText(job.schedule_display) || asText(job.schedule?.display) || ''

  // Stuck flag on the row uses only the job's own timing signal; the full
  // last-active check happens in the detail (which has the run session).
  const stuckByJob = taskPhase(job) === 'running' && Boolean(job.last_run_at) && isStuck(job, null, now)

  if (stuckByJob) {
    return <span className="truncate text-[0.7rem] text-amber-600 dark:text-amber-300">{t.tasks.stuckHint}</span>
  }

  return <span className="truncate text-[0.7rem] text-muted-foreground">{display || t.tasks.pending}</span>
}

// ── Detail: header + live progress card + run history ────────────────────────

function TaskDetail({
  busy,
  c,
  job,
  now,
  onDelete,
  onOpenSession,
  onRunNow
}: {
  busy: boolean
  c: Translations['tasks']
  job: CronJob
  now: number
  onDelete: () => void
  onOpenSession?: (sessionId: string) => void
  onRunNow: () => void
}) {
  const phase = taskPhase(job)
  const prompt = jobPrompt(job)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-6 px-6 py-6">
          <header className="space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-xl font-semibold tracking-tight">{jobTitle(job)}</h3>
                  <PhasePill c={c} phase={phase} />
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.7rem] text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="size-3" />
                    {asText(job.schedule_display) || asText(job.schedule?.display) || c.pending}
                  </span>
                  <span>
                    {c.started} {formatTime(job.last_run_at)}
                  </span>
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {phase !== 'running' && (
                  <Button disabled={busy} onClick={onRunNow} size="sm" variant="outline">
                    <Codicon name="zap" size="0.875rem" />
                    {c.runAgain}
                  </Button>
                )}
                <Button
                  className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  onClick={onDelete}
                  size="sm"
                  variant="ghost"
                >
                  <Codicon name="trash" size="0.875rem" />
                </Button>
              </div>
            </div>

            {prompt && (
              <div className="rounded-md bg-(--ui-bg-quinary) px-3 py-2">
                <div className="mb-1 text-[0.62rem] font-medium uppercase tracking-wide text-muted-foreground">
                  {c.goalLabel}
                </div>
                <p className="whitespace-pre-wrap text-xs text-foreground/90">{prompt}</p>
              </div>
            )}
            {job.last_error && (
              <p className="inline-flex items-start gap-1 text-[0.7rem] text-destructive">
                <AlertTriangle className="mt-px size-3 shrink-0" />
                <span className="line-clamp-3">{job.last_error}</span>
              </p>
            )}
          </header>

          <TaskProgressPanel c={c} job={job} now={now} onOpenSession={onOpenSession} />
        </div>
      </div>
    </div>
  )
}

// Poll the run + its transcript while the detail is open so an in-flight task's
// progress advances without a manual reload. Same cadence as the cron runs poll.
const PROGRESS_POLL_MS = 8000

function TaskProgressPanel({
  c,
  job,
  now,
  onOpenSession
}: {
  c: Translations['tasks']
  job: CronJob
  now: number
  onOpenSession?: (sessionId: string) => void
}) {
  const [runs, setRuns] = useState<null | SessionInfo[]>(null)
  const [progress, setProgress] = useState<null | TaskProgress>(null)
  const seenRunRef = useRef<null | string>(null)

  useEffect(() => {
    let cancelled = false
    seenRunRef.current = null
    setRuns(null)
    setProgress(null)

    const load = async () => {
      try {
        const fetched = await getCronJobRuns(job.id)

        if (cancelled) {
          return
        }

        setRuns(fetched)

        const run = primaryRun(fetched)

        if (!run) {
          setProgress(null)

          return
        }

        // Re-fetch the transcript on every poll for the active run so live todos
        // advance; for a finished run one fetch is enough but re-fetching is cheap.
        const { messages } = await getSessionMessages(run.id)

        if (!cancelled) {
          setProgress(deriveProgress(messages as SessionMessage[]))
          seenRunRef.current = run.id
        }
      } catch {
        if (!cancelled) {
          setRuns(prev => prev ?? [])
        }
      }
    }

    void load()

    const intervalId = window.setInterval(() => {
      if (document.visibilityState === 'visible' && taskPhase(job) === 'running') {
        void load()
      }
    }, PROGRESS_POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [job.id, job.state, job.enabled])

  const run = runs ? primaryRun(runs) : null
  const stuck = isStuck(job, run, now)
  const phase = taskPhase(job)

  return (
    <div className="space-y-4">
      {stuck && (
        <div className="flex items-start gap-2 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{c.stuckDetail}</span>
        </div>
      )}

      {phase === 'running' && !run && (
        <div className="flex items-center gap-2 rounded-md bg-(--ui-bg-quinary) px-3 py-2 text-xs text-muted-foreground">
          <Codicon name="loading" size="0.75rem" spinning />
          {c.waitingToStart}
        </div>
      )}

      {progress && (progress.totalSteps > 0 || progress.latestOutput) && (
        <ProgressCard c={c} progress={progress} running={phase === 'running'} />
      )}

      <TaskRuns c={c} onOpenSession={onOpenSession} runs={runs} />
    </div>
  )
}

function ProgressCard({
  c,
  progress,
  running
}: {
  c: Translations['tasks']
  progress: TaskProgress
  running: boolean
}) {
  const { completedSteps, currentStep, latestOutput, totalSteps } = progress
  const pct = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0

  return (
    <div className="rounded-lg border border-(--stroke-nous) bg-popover/60 p-3.5">
      {totalSteps > 0 && (
        <>
          <div className="mb-1.5 flex items-center justify-between text-[0.7rem]">
            <span className="font-medium text-foreground">{c.progressLabel}</span>
            <span className="tabular-nums text-muted-foreground">{c.stepsOf(completedSteps, totalSteps)}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn('h-full rounded-full bg-primary transition-[width] duration-500', running && 'opacity-90')}
              style={{ width: `${pct}%` }}
            />
          </div>
          {currentStep && (
            <div className="mt-2 flex items-start gap-1.5 text-xs text-foreground/90">
              {running && <Codicon className="mt-0.5 shrink-0 text-primary" name="loading" size="0.75rem" spinning />}
              <span className="min-w-0">
                <span className="text-muted-foreground">{c.currentStepLabel} </span>
                {currentStep}
              </span>
            </div>
          )}
        </>
      )}

      {latestOutput && (
        <div className={cn(totalSteps > 0 && 'mt-3 border-t border-(--stroke-nous) pt-3')}>
          <div className="mb-1 text-[0.62rem] font-medium uppercase tracking-wide text-muted-foreground">
            {c.latestOutputLabel}
          </div>
          <p className="line-clamp-4 whitespace-pre-wrap text-xs text-foreground/90">{latestOutput}</p>
        </div>
      )}
    </div>
  )
}

function TaskRuns({
  c,
  onOpenSession,
  runs
}: {
  c: Translations['tasks']
  onOpenSession?: (sessionId: string) => void
  runs: null | SessionInfo[]
}) {
  return (
    <div>
      <div className="mb-1.5 text-[0.62rem] font-medium uppercase tracking-wide text-muted-foreground">
        {c.runHistory}
        {runs && runs.length > 0 ? ` · ${runs.length}` : ''}
      </div>
      {runs === null ? (
        <div className="flex items-center gap-1.5 py-1 text-xs text-muted-foreground">
          <Codicon name="loading" size="0.75rem" spinning />
        </div>
      ) : runs.length === 0 ? (
        <div className="py-1 text-xs text-muted-foreground">{c.noRuns}</div>
      ) : (
        <div className="flex flex-col gap-px">
          {runs.map(run => (
            <button
              className="flex items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs hover:bg-(--chrome-action-hover) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              key={run.id}
              onClick={() => onOpenSession?.(run.id)}
              type="button"
            >
              <span className="inline-flex min-w-0 items-center gap-1.5">
                {run.is_active && <span className="size-1.5 shrink-0 rounded-full bg-primary motion-safe:animate-pulse" />}
                <span className="truncate text-foreground">{run.title?.trim() || run.preview?.trim() || run.id}</span>
              </span>
              <span className="shrink-0 text-[0.62rem] text-muted-foreground tabular-nums">
                {formatEpoch(run.last_active || run.started_at)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function PhasePill({ c, phase }: { c: Translations['tasks']; phase: TaskPhase }) {
  const tone =
    phase === 'failed'
      ? 'bg-destructive/10 text-destructive'
      : phase === 'done'
        ? 'bg-muted text-muted-foreground'
        : 'bg-primary/10 text-primary'

  return (
    <span className={cn('inline-flex items-center rounded-full px-1.5 py-0.5 text-[0.64rem]', tone)}>
      {c.phases[phase]}
    </span>
  )
}

// ── Create dialog ────────────────────────────────────────────────────────────

type WhenKind = 'now' | 'in' | 'at'

const DELAY_PRESETS: readonly string[] = ['30m', '1h', '2h', '6h', '1d']

function TaskEditorDialog({
  c,
  commonCopy,
  onClose,
  onCreate,
  open,
  submitting
}: {
  c: Translations['tasks']
  commonCopy: Translations['common']
  onClose: () => void
  onCreate: (prompt: string, when: TaskWhen) => Promise<void>
  open: boolean
  submitting: boolean
}) {
  const [prompt, setPrompt] = useState('')
  const [whenKind, setWhenKind] = useState<WhenKind>('now')
  const [delay, setDelay] = useState('1h')
  const [at, setAt] = useState('')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setPrompt('')
    setWhenKind('now')
    setDelay('1h')
    setAt('')
    setError(null)
  }, [open])

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const trimmed = prompt.trim()

    if (!trimmed) {
      setError(c.goalRequired)

      return
    }

    let when: TaskWhen

    if (whenKind === 'now') {
      when = { kind: 'now' }
    } else if (whenKind === 'in') {
      when = { kind: 'in', value: delay }
    } else {
      if (!at.trim()) {
        setError(c.timeRequired)

        return
      }

      when = { kind: 'at', value: at }
    }

    setError(null)

    try {
      await onCreate(trimmed, when)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : c.failedStart)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !submitting && onClose()} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{c.newTaskTitle}</DialogTitle>
          <DialogDescription>{c.newTaskDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium text-foreground" htmlFor="task-goal">
              {c.goalLabel}
            </label>
            <Textarea
              autoFocus
              className="min-h-28"
              id="task-goal"
              onChange={event => setPrompt(event.target.value)}
              placeholder={c.goalPlaceholder}
              value={prompt}
            />
          </div>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <label className="text-xs font-medium text-foreground" htmlFor="task-when">
                {c.whenLabel}
              </label>
              <Select onValueChange={value => setWhenKind(value as WhenKind)} value={whenKind}>
                <SelectTrigger className="h-9 rounded-md" id="task-when">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="now">{c.whenNow}</SelectItem>
                  <SelectItem value="in">{c.whenIn}</SelectItem>
                  <SelectItem value="at">{c.whenAt}</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {whenKind === 'in' && (
              <div className="grid gap-1.5">
                <label className="text-xs font-medium text-foreground" htmlFor="task-delay">
                  {c.delayLabel}
                </label>
                <Select onValueChange={setDelay} value={delay}>
                  <SelectTrigger className="h-9 rounded-md" id="task-delay">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DELAY_PRESETS.map(value => (
                      <SelectItem key={value} value={value}>
                        {value}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {whenKind === 'at' && (
              <div className="grid gap-1.5">
                <label className="text-xs font-medium text-foreground" htmlFor="task-at">
                  {c.atLabel}
                </label>
                <Input
                  className="font-mono"
                  id="task-at"
                  onChange={event => setAt(event.target.value)}
                  placeholder="2026-02-03T14:00"
                  type="datetime-local"
                  value={at}
                />
              </div>
            )}
          </div>

          <p className="text-[0.66rem] leading-4 text-muted-foreground">{c.persistNote}</p>

          {error && (
            <div className="flex items-start gap-2 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={submitting} onClick={onClose} type="button" variant="outline">
              {commonCopy.cancel}
            </Button>
            <Button disabled={submitting} type="submit">
              {submitting ? commonCopy.saving : c.startTask}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
