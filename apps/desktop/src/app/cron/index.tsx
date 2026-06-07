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
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import {
  createCronJob,
  type CronJob,
  deleteCronJob,
  getCronJobRuns,
  getCronJobs,
  pauseCronJob,
  resumeCronJob,
  type SessionInfo,
  triggerCronJob,
  updateCronJob
} from '@/hermes'
import { AlertTriangle, Clock, Pause, Pencil, Play, Trash2, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { useTranslation } from '@/hooks/use-translation'
import { t } from '@/store/i18n'
import { PageSearchShell } from '../page-search-shell'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

const DEFAULT_DELIVER = 'local'

const DELIVERY_VALUES: readonly string[] = ['local', 'telegram', 'discord', 'slack', 'email']

const SCHEDULE_OPTIONS: ReadonlyArray<ScheduleOption> = [
  { expr: '0 9 * * *', value: 'daily' },
  { expr: '0 9 * * 1-5', value: 'weekdays' },
  { expr: '0 9 * * 1', value: 'weekly' },
  { expr: '0 9 1 * *', value: 'monthly' },
  { expr: '0 * * * *', value: 'hourly' },
  { expr: '*/15 * * * *', value: 'every-15-minutes' },
  { value: 'custom' }
]

const STATE_TONE: Record<string, 'good' | 'muted' | 'warn' | 'bad'> = {
  enabled: 'good',
  scheduled: 'good',
  running: 'good',
  paused: 'warn',
  disabled: 'muted',
  error: 'bad',
  completed: 'muted'
}

const PILL_TONE: Record<'good' | 'muted' | 'warn' | 'bad', string> = {
  good: 'bg-primary/10 text-primary',
  muted: 'bg-muted text-muted-foreground',
  warn: 'bg-amber-500/10 text-amber-600 dark:text-amber-300',
  bad: 'bg-destructive/10 text-destructive'
}

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

const truncate = (value: string, max = 80): string => (value.length > max ? `${value.slice(0, max)}…` : value)

function jobName(job: CronJob): string {
  return asText(job.name).trim()
}

function jobPrompt(job: CronJob): string {
  return asText(job.prompt)
}

function jobScheduleDisplay(job: CronJob): string {
  return asText(job.schedule_display) || asText(job.schedule?.display) || asText(job.schedule?.expr) || '—'
}

function jobScheduleExpr(job: CronJob): string {
  return asText(job.schedule?.expr) || asText(job.schedule_display) || ''
}

function jobDeliver(job: CronJob): string {
  return asText(job.deliver) || DEFAULT_DELIVER
}

function cronParts(expr: string): null | string[] {
  const parts = expr.trim().replace(/\s+/g, ' ').split(' ')

  return parts.length === 5 ? parts : null
}

function dayName(value: string, c: Translations['cron']): string {
  return c.days[value] ?? c.dayFallback(value)
}

function formatCronTime(minute: string, hour: string): string {
  const numericHour = Number(hour)
  const numericMinute = Number(minute)

  if (!Number.isInteger(numericHour) || !Number.isInteger(numericMinute)) {
    return `${hour}:${minute}`
  }

  return new Date(2000, 0, 1, numericHour, numericMinute).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  })
}

function isIntegerToken(value: string): boolean {
  return /^\d+$/.test(value)
}

function scheduleOptionForExpr(expr: string): ScheduleOption {
  const normalized = expr.trim().replace(/\s+/g, ' ')
  const exactMatch = SCHEDULE_OPTIONS.find(option => option.expr === normalized)

  if (exactMatch) {
    return exactMatch
  }

  const parts = cronParts(normalized)

  if (!parts) {
    return SCHEDULE_OPTIONS[SCHEDULE_OPTIONS.length - 1]
  }

  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts

  if (dayOfMonth === '*' && month === '*' && dayOfWeek === '*' && isIntegerToken(minute) && isIntegerToken(hour)) {
    return SCHEDULE_OPTIONS.find(option => option.value === 'daily') ?? SCHEDULE_OPTIONS[0]
  }

  if (dayOfMonth === '*' && month === '*' && dayOfWeek === '1-5' && isIntegerToken(minute) && isIntegerToken(hour)) {
    return SCHEDULE_OPTIONS.find(option => option.value === 'weekdays') ?? SCHEDULE_OPTIONS[0]
  }

  if (
    dayOfMonth === '*' &&
    month === '*' &&
    isIntegerToken(dayOfWeek) &&
    isIntegerToken(minute) &&
    isIntegerToken(hour)
  ) {
    return SCHEDULE_OPTIONS.find(option => option.value === 'weekly') ?? SCHEDULE_OPTIONS[0]
  }

  if (
    month === '*' &&
    dayOfWeek === '*' &&
    isIntegerToken(dayOfMonth) &&
    isIntegerToken(minute) &&
    isIntegerToken(hour)
  ) {
    return SCHEDULE_OPTIONS.find(option => option.value === 'monthly') ?? SCHEDULE_OPTIONS[0]
  }

  if (hour === '*' && dayOfMonth === '*' && month === '*' && dayOfWeek === '*' && isIntegerToken(minute)) {
    return SCHEDULE_OPTIONS.find(option => option.value === 'hourly') ?? SCHEDULE_OPTIONS[0]
  }

  if (normalized === '*/15 * * * *') {
    return SCHEDULE_OPTIONS.find(option => option.value === 'every-15-minutes') ?? SCHEDULE_OPTIONS[0]
  }

  return SCHEDULE_OPTIONS[SCHEDULE_OPTIONS.length - 1]
}

function scheduleSummary(option: ScheduleOption, expr: string, c: Translations['cron']): string {
  const parts = cronParts(expr)

  if (!parts) {
    return c.scheduleHints[option.value] ?? ''
  }

  const [minute, hour, dayOfMonth, , dayOfWeek] = parts

  if (option.value === 'daily') {
    return c.everyDayAt(formatCronTime(minute, hour))
  }

  if (option.value === 'weekdays') {
    return c.weekdaysAt(formatCronTime(minute, hour))
  }

  if (option.value === 'weekly') {
    return c.everyDayOfWeekAt(dayName(dayOfWeek, c), formatCronTime(minute, hour))
  }

  if (option.value === 'monthly') {
    return c.monthlyOnDayAt(dayOfMonth, formatCronTime(minute, hour))
  }

  if (option.value === 'hourly') {
    return minute === '0' ? c.topOfHour : c.everyHourAt(minute.padStart(2, '0'))
  }

  return c.scheduleHints[option.value] ?? ''
}

function formatTime(iso?: null | string): string {
  if (!iso) {
    return '—'
  }

  const date = new Date(iso)

  if (Number.isNaN(date.valueOf())) {
    return iso
  }

  return date.toLocaleString()
}

function matchesQuery(job: CronJob, q: string): boolean {
  if (!q) {
    return true
  }

  const needle = q.toLowerCase()

  return [jobTitle(job), jobPrompt(job), jobScheduleDisplay(job), jobScheduleExpr(job), jobDeliver(job)].some(value =>
    value.toLowerCase().includes(needle)
  )
}

interface CronViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function CronView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: CronViewProps) {
  const { t } = useTranslation()
  const [jobs, setJobs] = useState<CronJob[] | null>(null)
  const [query, setQuery] = useState('')
  const [busyJobId, setBusyJobId] = useState<null | string>(null)
  // Master/detail: the job whose schedule + run history fill the right pane.
  const [selectedJobId, setSelectedJobId] = useState<null | string>(null)
  // Set when a job is opened from the sidebar so we scroll it into view once the
  // row exists. Cleared after the scroll fires.
  const pendingScrollRef = useRef<null | string>(null)
  const focusJobId = useStore($cronFocusJobId)

  const [editor, setEditor] = useState<EditorState>({ mode: 'closed' })
  const [pendingDelete, setPendingDelete] = useState<CronJob | null>(null)
  const [deleting, setDeleting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setCronJobs(await getCronJobs())
    } catch (err) {
      notifyError(err, 'Failed to load cron jobs')
    } finally {
      setRefreshing(false)
    }
  }, [c])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Sidebar → "open this job": resolve the focus id (or name) to a job, select
  // it, queue a scroll, then clear the one-shot focus so re-opening cron
  // normally doesn't re-trigger it.
  useEffect(() => {
    if (!focusJobId) {return}

    const match = jobs.find(job => job.id === focusJobId || jobName(job) === focusJobId)

    if (match) {
      setSelectedJobId(match.id)
      pendingScrollRef.current = match.id
    }

    setCronFocusJobId(null)
  }, [focusJobId, jobs])

  const visibleJobs = useMemo(
    () => jobs.filter(job => matchesQuery(job, query.trim())).sort((a, b) => jobTitle(a).localeCompare(jobTitle(b))),
    [jobs, query]
  )

  // Detail always reflects a concrete job: the explicitly selected one, else the
  // first visible row, so the right pane is never empty while jobs exist.
  const selectedJob = useMemo(
    () => visibleJobs.find(job => job.id === selectedJobId) ?? visibleJobs[0] ?? null,
    [visibleJobs, selectedJobId]
  )

  // Scroll a sidebar-opened job into view once its list row is mounted.
  useEffect(() => {
    const target = pendingScrollRef.current

    if (!target || selectedJob?.id !== target) {return}

    pendingScrollRef.current = null
    requestAnimationFrame(() => {
      document.querySelector(`[data-cron-row="${CSS.escape(target)}"]`)?.scrollIntoView({ block: 'nearest' })
    })
  }, [selectedJob])

  const totalCount = jobs.length

  async function handlePauseResume(job: CronJob) {
    setBusyJobId(job.id)

    try {
      const isPaused = jobState(job) === 'paused'
      const updated = isPaused ? await resumeCronJob(job.id) : await pauseCronJob(job.id)
      updateCronJobs(rows => rows.map(row => (row.id === job.id ? updated : row)))
      notify({
        kind: 'success',
        title: isPaused ? c.resumed : c.paused,
        message: truncate(jobTitle(job), 60)
      })
    } catch (err) {
      notifyError(err, c.failedUpdate)
    } finally {
      setBusyJobId(null)
    }
  }

  async function handleTrigger(job: CronJob) {
    setBusyJobId(job.id)

    try {
      const updated = await triggerCronJob(job.id)
      setJobs(current => (current ? current.map(row => (row.id === job.id ? updated : row)) : current))
      notify({ kind: 'success', title: c.triggered, message: truncate(jobTitle(job), 60) })
    } catch (err) {
      notifyError(err, c.failedTrigger)
    } finally {
      setBusyJobId(null)
    }
  }

  async function handleConfirmDelete() {
    if (!pendingDelete) {
      return
    }

    setDeleting(true)

    try {
      await deleteCronJob(pendingDelete.id)
      setJobs(current => (current ? current.filter(row => row.id !== pendingDelete.id) : current))
      notify({ kind: 'success', title: c.deleted, message: truncate(jobTitle(pendingDelete), 60) })
      setPendingDelete(null)
    } catch (err) {
      notifyError(err, c.failedDelete)
    } finally {
      setDeleting(false)
    }
  }

  async function handleEditorSave(values: EditorValues) {
    if (editor.mode === 'create') {
      const created = await createCronJob({
        prompt: values.prompt,
        schedule: values.schedule,
        name: values.name || undefined,
        deliver: values.deliver || DEFAULT_DELIVER
      })

      setJobs(current => (current ? [...current, created] : [created]))
      notify({ kind: 'success', title: c.created, message: truncate(jobTitle(created), 60) })
    } else if (editor.mode === 'edit') {
      const updated = await updateCronJob(editor.job.id, {
        prompt: values.prompt,
        schedule: values.schedule,
        name: values.name,
        deliver: values.deliver
      })

      setJobs(current => (current ? current.map(row => (row.id === updated.id ? updated : row)) : current))
      notify({ kind: 'success', title: c.updated, message: truncate(jobTitle(updated), 60) })
    }

    setEditor({ mode: 'closed' })
  }

  return (
    <PageSearchShell
      {...props}
      onSearchChange={setQuery}
      searchPlaceholder="Search cron jobs..."
      searchTrailingAction={
        <Button
          aria-label={refreshing ? 'Refreshing cron jobs' : 'Refresh cron jobs'}
          className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
          disabled={refreshing}
          onClick={() => void refresh()}
          size="icon-xs"
          title={refreshing ? 'Refreshing cron jobs' : 'Refresh cron jobs'}
          type="button"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.875rem" spinning={refreshing} />
        </Button>
      }
      searchValue={query}
    >
      {!jobs ? (
        <PageLoader label="Loading cron jobs..." />
      ) : visibleJobs.length === 0 ? (
        // Empty state owns the primary "create" CTA — we used to also have
        // one in the filters bar but it was redundant. Only show the button
        // when there are zero jobs total; the search-empty case ("No
        // matches") just asks the user to broaden their query.
        <EmptyState
          actionLabel={totalCount === 0 ? 'Create first cron' : undefined}
          description={
            totalCount === 0
              ? 'Schedule a prompt to run on a cron expression. Hermes will run it and deliver results to the destination you pick.'
              : 'Try a broader search query.'
          }
          onAction={totalCount === 0 ? () => setEditor({ mode: 'create' }) : undefined}
          title={totalCount === 0 ? t('cron.noJobs') : 'No matches'}
        />
      ) : (
        <OverlaySplitLayout>
          <OverlaySidebar>
            <OverlayNewButton label={c.newCron} onClick={() => setEditor({ mode: 'create' })} />
            {totalCount > 0 && (
              <SearchField
                aria-label={c.search}
                containerClassName="mb-1 w-full px-2"
                onChange={setQuery}
                placeholder={c.search}
                value={query}
              />
            )}
            {visibleJobs.map(job => (
              <CronJobListRow
                active={selectedJob?.id === job.id}
                c={c}
                job={job}
                key={job.id}
                onSelect={() => setSelectedJobId(job.id)}
              />
            ))}
            {visibleJobs.length === 0 && (
              <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                {totalCount === 0 ? c.emptyTitleNew : c.emptyTitleSearch}
              </p>
            )}
          </OverlaySidebar>

      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t('cron.deleteCronJob')}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? (
                <>
                  This will remove{' '}
                  <span className="font-medium text-foreground">{truncate(jobTitle(pendingDelete), 60)}</span>{' '}
                  permanently. It will stop firing immediately.
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} variant="outline">
              Cancel
            </Button>
            <Button disabled={deleting} onClick={() => void handleConfirmDelete()} variant="destructive">
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </PageSearchShell>
  )
}

function CronJobRow({
  busy,
  c,
  job,
  onDelete,
  onEdit,
  onOpenSession,
  onPauseResume,
  onTrigger
}: {
  busy: boolean
  c: Translations['cron']
  job: CronJob
  onDelete: () => void
  onEdit: () => void
  onOpenSession?: (sessionId: string) => void
  onPauseResume: () => void
  onTrigger: () => void
}) {
  const state = jobState(job)
  const isPaused = state === 'paused'
  const deliver = jobDeliver(job)
  const prompt = jobPrompt(job)

  return (
    <div className="grid gap-3 px-3 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
      <button
        className="min-w-0 cursor-pointer rounded-md text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        onClick={onEdit}
        type="button"
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium">{jobTitle(job)}</span>
          <StatePill tone={STATE_TONE[state] ?? 'muted'}>{state}</StatePill>
          {deliver && deliver !== DEFAULT_DELIVER && <StatePill tone="muted">{deliver}</StatePill>}
        </div>
        {hasName && prompt && <p className="mt-1 truncate text-xs text-muted-foreground">{truncate(prompt, 120)}</p>}
        <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.68rem] text-muted-foreground">
          <span className="inline-flex items-center gap-1 font-mono">
            <Clock className="size-3" />
            {jobScheduleDisplay(job)}
          </span>
          <span>
            {c.last} {formatTime(job.last_run_at)}
          </span>
          <span>
            {c.next} {formatTime(job.next_run_at)}
          </span>
        </div>
        {job.last_error && (
          <p className="mt-1 inline-flex items-start gap-1 text-[0.68rem] text-destructive">
            <AlertTriangle className="mt-px size-3 shrink-0" />
            <span className="line-clamp-2">{job.last_error}</span>
          </p>
        )}
      </button>

      <div className="flex shrink-0 items-center">
        <CronJobActionsMenu
          busy={busy}
          isPaused={isPaused}
          onDelete={onDelete}
          onEdit={onEdit}
          onPauseResume={onPauseResume}
          onTrigger={onTrigger}
          title={jobTitle(job)}
        >
          {isPaused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
        </IconAction>
        <IconAction aria-label="Trigger now" disabled={busy} onClick={onTrigger} title="Trigger now">
          <Zap className="size-3.5" />
        </IconAction>
        <IconAction aria-label="Edit cron" onClick={onEdit} title="Edit">
          <Pencil className="size-3.5" />
        </IconAction>
        <IconAction
          aria-label="Delete cron"
          className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          onClick={onDelete}
          title="Delete"
        >
          <Trash2 className="size-3.5" />
        </IconAction>
      </div>
    </div>
  )
}

function StatePill({ children, tone }: { children: string; tone: keyof typeof PILL_TONE }) {
  return (
    <Button
      className={cn('size-7 text-muted-foreground hover:text-foreground', className)}
      size="icon"
      variant="ghost"
      {...props}
    >
      {children}
    </span>
  )
}

function StatePill({ children, tone }: { children: string; tone: keyof typeof PILL_TONE }) {
  return (
    <span
      className={cn('inline-flex items-center rounded-full px-1.5 py-0.5 text-[0.64rem] capitalize', PILL_TONE[tone])}
    >
      {children}
    </span>
  )
}

function CronEditorDialog({
  editor,
  onClose,
  onSave
}: {
  editor: EditorState
  onClose: () => void
  onSave: (values: EditorValues) => Promise<void>
}) {
  const { t } = useI18n()
  const c = t.cron
  const open = editor.mode !== 'closed'
  const isEdit = editor.mode === 'edit'
  const initial = isEdit ? editor.job : null

  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')
  const [schedule, setSchedule] = useState('')
  const [schedulePreset, setSchedulePreset] = useState('daily')
  const [deliver, setDeliver] = useState(DEFAULT_DELIVER)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName(initial ? jobName(initial) : '')
    setPrompt(initial ? jobPrompt(initial) : '')
    setSchedule(initial ? jobScheduleExpr(initial) : (SCHEDULE_OPTIONS[0].expr ?? ''))
    setSchedulePreset(initial ? scheduleOptionForExpr(jobScheduleExpr(initial)).value : 'daily')
    setDeliver(initial ? jobDeliver(initial) : DEFAULT_DELIVER)
    setError(null)
    setSaving(false)
  }, [initial, open])

  const selectedScheduleOption =
    SCHEDULE_OPTIONS.find(candidate => candidate.value === schedulePreset) ?? SCHEDULE_OPTIONS[0]

  function handleSchedulePresetChange(nextPreset: string) {
    setSchedulePreset(nextPreset)
    setError(null)

    const option = SCHEDULE_OPTIONS.find(candidate => candidate.value === nextPreset)

    if (option?.expr) {
      setSchedule(option.expr)
    } else if (scheduleOptionForExpr(schedule).value !== 'custom') {
      setSchedule('')
    }
  }

  const scheduleHint = scheduleSummary(selectedScheduleOption, schedule, c)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const trimmedPrompt = prompt.trim()
    const trimmedSchedule = schedule.trim()

    if (!trimmedPrompt || !trimmedSchedule) {
      setError(c.promptScheduleRequired)

      return
    }

    setSaving(true)
    setError(null)

    try {
      await onSave({
        deliver,
        name: name.trim(),
        prompt: trimmedPrompt,
        schedule: trimmedSchedule
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : c.failedSave)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{isEdit ? c.editTitle : c.createTitle}</DialogTitle>
          <DialogDescription>{isEdit ? c.editDesc : c.createDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <Field htmlFor="cron-name" label={c.nameLabel} optional optionalLabel={c.optional}>
            <Input
              autoFocus
              id="cron-name"
              onChange={event => setName(event.target.value)}
              placeholder={c.namePlaceholder}
              value={name}
            />
          </Field>

          <Field htmlFor="cron-prompt" label={c.promptLabel}>
            <Textarea
              className="min-h-24 font-mono"
              id="cron-prompt"
              onChange={event => setPrompt(event.target.value)}
              placeholder={c.promptPlaceholder}
              value={prompt}
            />
          </Field>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="cron-frequency" label={c.frequencyLabel}>
              <Select onValueChange={handleSchedulePresetChange} value={schedulePreset}>
                <SelectTrigger className="h-9 rounded-md" id="cron-frequency">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SCHEDULE_OPTIONS.map(option => (
                    <SelectItem key={option.value} value={option.value}>
                      {c.scheduleLabels[option.value]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field htmlFor="cron-deliver" label={c.deliverLabel}>
              <Select onValueChange={setDeliver} value={deliver}>
                <SelectTrigger className="h-9 rounded-md" id="cron-deliver">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DELIVERY_VALUES.map(value => (
                    <SelectItem key={value} value={value}>
                      {c.deliveryLabels[value]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          {schedulePreset === 'custom' ? (
            <Field htmlFor="cron-schedule" label={c.customScheduleLabel}>
              <Input
                className="font-mono"
                id="cron-schedule"
                onChange={event => setSchedule(event.target.value)}
                placeholder={c.customPlaceholder}
                value={schedule}
              />
              <FieldHint>{c.customHint}</FieldHint>
            </Field>
          ) : (
            <div className="rounded-md bg-(--ui-bg-quinary) px-3 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-medium text-foreground">{scheduleHint}</span>
                <span className="font-mono text-muted-foreground">{schedule}</span>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={saving} type="submit">
              {saving ? t.common.saving : isEdit ? c.saveChanges : c.createAction}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function Field({
  children,
  htmlFor,
  label,
  optional,
  optionalLabel
}: {
  children: React.ReactNode
  htmlFor: string
  label: string
  optional?: boolean
  optionalLabel?: string
}) {
  return (
    <div className="grid gap-1.5">
      <label className="flex items-baseline gap-2 text-xs font-medium text-foreground" htmlFor={htmlFor}>
        {label}
        {optional && <span className="text-[0.65rem] font-normal text-muted-foreground">{t('cron.optional')}</span>}
      </label>
      {children}
    </div>
  )
}

function FieldHint({ children }: { children: React.ReactNode }) {
  return <p className="text-[0.66rem] leading-4 text-muted-foreground">{children}</p>
}

type EditorState = { mode: 'closed' } | { mode: 'create' } | { job: CronJob; mode: 'edit' }

interface EditorValues {
  deliver: string
  name: string
  prompt: string
  schedule: string
}

interface ScheduleOption {
  expr?: string
  value: string
}
