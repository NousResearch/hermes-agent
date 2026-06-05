import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge, type BadgeProps } from '@/components/ui/badge'
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
  getCronJobs,
  pauseCronJob,
  resumeCronJob,
  triggerCronJob,
  updateCronJob
} from '@/hermes'
import { type Translate, useTranslation } from '@/i18n'
import { AlertTriangle, Clock } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayView } from '../overlays/overlay-view'

import { CronJobActionsMenu, CronJobActionsTrigger } from './cron-job-actions-menu'

const DEFAULT_DELIVER = 'local'

const DELIVERY_OPTIONS: ReadonlyArray<{ labelKey: string; value: string }> = [
  { labelKey: 'cron.delivery.local', value: 'local' },
  { labelKey: 'cron.delivery.telegram', value: 'telegram' },
  { labelKey: 'cron.delivery.discord', value: 'discord' },
  { labelKey: 'cron.delivery.slack', value: 'slack' },
  { labelKey: 'cron.delivery.email', value: 'email' }
]

const SCHEDULE_OPTIONS: ReadonlyArray<ScheduleOption> = [
  {
    expr: '0 9 * * *',
    hintKey: 'cron.schedule.dailyHint',
    labelKey: 'cron.schedule.daily',
    value: 'daily'
  },
  {
    expr: '0 9 * * 1-5',
    hintKey: 'cron.schedule.weekdaysHint',
    labelKey: 'cron.schedule.weekdays',
    value: 'weekdays'
  },
  {
    expr: '0 9 * * 1',
    hintKey: 'cron.schedule.weeklyHint',
    labelKey: 'cron.schedule.weekly',
    value: 'weekly'
  },
  {
    expr: '0 9 1 * *',
    hintKey: 'cron.schedule.monthlyHint',
    labelKey: 'cron.schedule.monthly',
    value: 'monthly'
  },
  {
    expr: '0 * * * *',
    hintKey: 'cron.schedule.hourlyHint',
    labelKey: 'cron.schedule.hourly',
    value: 'hourly'
  },
  {
    expr: '*/15 * * * *',
    hintKey: 'cron.schedule.every15MinutesHint',
    labelKey: 'cron.schedule.every15Minutes',
    value: 'every-15-minutes'
  },
  {
    hintKey: 'cron.schedule.customHint',
    labelKey: 'cron.schedule.custom',
    value: 'custom'
  }
]

const STATE_VARIANT: Record<string, BadgeProps['variant']> = {
  enabled: 'default',
  scheduled: 'default',
  running: 'default',
  paused: 'warn',
  disabled: 'muted',
  error: 'destructive',
  completed: 'muted'
}

const STATE_LABEL_KEYS: Record<string, string> = {
  completed: 'cron.states.completed',
  disabled: 'cron.states.disabled',
  enabled: 'cron.states.enabled',
  error: 'cron.states.error',
  paused: 'cron.states.paused',
  running: 'cron.states.running',
  scheduled: 'cron.states.scheduled'
}

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

const truncate = (value: string, max = 80): string => (value.length > max ? `${value.slice(0, max)}…` : value)

function jobName(job: CronJob): string {
  return asText(job.name).trim()
}

function jobPrompt(job: CronJob): string {
  return asText(job.prompt)
}

function jobTitle(job: CronJob): string {
  const name = jobName(job)

  if (name) {
    return name
  }

  const prompt = jobPrompt(job)

  if (prompt) {
    return truncate(prompt, 60)
  }

  const script = asText(job.script)

  if (script) {
    return truncate(script, 60)
  }

  return job.id || 'Cron job'
}

function jobScheduleDisplay(job: CronJob): string {
  return asText(job.schedule_display) || asText(job.schedule?.display) || asText(job.schedule?.expr) || '—'
}

function jobScheduleExpr(job: CronJob): string {
  return asText(job.schedule?.expr) || asText(job.schedule_display) || ''
}

function jobState(job: CronJob): string {
  return asText(job.state) || (job.enabled === false ? 'disabled' : 'scheduled')
}

function jobDeliver(job: CronJob): string {
  return asText(job.deliver) || DEFAULT_DELIVER
}

function cronParts(expr: string): null | string[] {
  const parts = expr.trim().replace(/\s+/g, ' ').split(' ')

  return parts.length === 5 ? parts : null
}

function dayName(value: string, t: Translate): string {
  const names: Record<string, string> = {
    '0': t('cron.days.sunday'),
    '1': t('cron.days.monday'),
    '2': t('cron.days.tuesday'),
    '3': t('cron.days.wednesday'),
    '4': t('cron.days.thursday'),
    '5': t('cron.days.friday'),
    '6': t('cron.days.saturday'),
    '7': t('cron.days.sunday')
  }

  return names[value] ?? t('cron.days.dayValue', { value })
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

function scheduleSummary(option: ScheduleOption, expr: string, t: Translate): string {
  const parts = cronParts(expr)

  if (!parts) {
    return t(option.hintKey)
  }

  const [minute, hour, dayOfMonth, , dayOfWeek] = parts

  if (option.value === 'daily') {
    return t('cron.summary.daily', { time: formatCronTime(minute, hour) })
  }

  if (option.value === 'weekdays') {
    return t('cron.summary.weekdays', { time: formatCronTime(minute, hour) })
  }

  if (option.value === 'weekly') {
    return t('cron.summary.weekly', { day: dayName(dayOfWeek, t), time: formatCronTime(minute, hour) })
  }

  if (option.value === 'monthly') {
    return t('cron.summary.monthly', { day: dayOfMonth, time: formatCronTime(minute, hour) })
  }

  if (option.value === 'hourly') {
    return minute === '0'
      ? t('cron.schedule.hourlyHint')
      : t('cron.summary.hourlyAtMinute', { minute: minute.padStart(2, '0') })
  }

  return t(option.hintKey)
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

interface CronViewProps {
  onClose: () => void
}

export function CronView({ onClose }: CronViewProps) {
  const t = useTranslation()
  const [jobs, setJobs] = useState<CronJob[] | null>(null)
  const [query, setQuery] = useState('')
  const [busyJobId, setBusyJobId] = useState<null | string>(null)

  const [editor, setEditor] = useState<EditorState>({ mode: 'closed' })
  const [pendingDelete, setPendingDelete] = useState<CronJob | null>(null)
  const [deleting, setDeleting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const result = await getCronJobs()
      setJobs(result)
    } catch (err) {
      notifyError(err, t('cron.notifications.loadFailed'))
    }
  }, [t])

  useRefreshHotkey(refresh)

  useEffect(() => {
    void refresh()
  }, [refresh])

  const visibleJobs = useMemo(() => {
    if (!jobs) {
      return []
    }

    return jobs.filter(job => matchesQuery(job, query.trim())).sort((a, b) => jobTitle(a).localeCompare(jobTitle(b)))
  }, [jobs, query])

  const enabledCount = jobs?.filter(job => job.enabled).length ?? 0
  const totalCount = jobs?.length ?? 0

  async function handlePauseResume(job: CronJob) {
    setBusyJobId(job.id)

    try {
      const isPaused = jobState(job) === 'paused'
      const updated = isPaused ? await resumeCronJob(job.id) : await pauseCronJob(job.id)
      setJobs(current => (current ? current.map(row => (row.id === job.id ? updated : row)) : current))
      notify({
        kind: 'success',
        title: isPaused ? t('cron.notifications.resumed') : t('cron.notifications.paused'),
        message: truncate(jobTitle(job), 60)
      })
    } catch (err) {
      notifyError(err, t('cron.notifications.updateFailed'))
    } finally {
      setBusyJobId(null)
    }
  }

  async function handleTrigger(job: CronJob) {
    setBusyJobId(job.id)

    try {
      const updated = await triggerCronJob(job.id)
      setJobs(current => (current ? current.map(row => (row.id === job.id ? updated : row)) : current))
      notify({ kind: 'success', title: t('cron.notifications.triggered'), message: truncate(jobTitle(job), 60) })
    } catch (err) {
      notifyError(err, t('cron.notifications.triggerFailed'))
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
      notify({
        kind: 'success',
        title: t('cron.notifications.deleted'),
        message: truncate(jobTitle(pendingDelete), 60)
      })
      setPendingDelete(null)
    } catch (err) {
      notifyError(err, t('cron.notifications.deleteFailed'))
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
      notify({ kind: 'success', title: t('cron.notifications.created'), message: truncate(jobTitle(created), 60) })
    } else if (editor.mode === 'edit') {
      const updated = await updateCronJob(editor.job.id, {
        prompt: values.prompt,
        schedule: values.schedule,
        name: values.name,
        deliver: values.deliver
      })

      setJobs(current => (current ? current.map(row => (row.id === updated.id ? updated : row)) : current))
      notify({ kind: 'success', title: t('cron.notifications.updated'), message: truncate(jobTitle(updated), 60) })
    }

    setEditor({ mode: 'closed' })
  }

  return (
    <OverlayView closeLabel={t('cron.close')} onClose={onClose}>
      <div className="flex min-h-0 flex-1 flex-col pt-[calc(var(--titlebar-height)+0.5rem)]">
        {totalCount > 0 && (
          <div className="mx-auto flex w-full max-w-4xl items-center gap-2 px-4 pb-2">
            <SearchField
              containerClassName="max-w-[60vw]"
              onChange={setQuery}
              placeholder={t('cron.search')}
              value={query}
            />
          </div>
        )}
        {!jobs ? (
          <PageLoader label={t('cron.loading')} />
        ) : visibleJobs.length === 0 ? (
          // Empty state owns the primary "create" CTA — we used to also have
          // one in the filters bar but it was redundant. Only show the button
          // when there are zero jobs total; the search-empty case ("No
          // matches") just asks the user to broaden their query.
          <EmptyState
            actionLabel={totalCount === 0 ? t('cron.empty.createFirst') : undefined}
            description={totalCount === 0 ? t('cron.empty.description') : t('cron.empty.searchDescription')}
            onAction={totalCount === 0 ? () => setEditor({ mode: 'create' }) : undefined}
            title={totalCount === 0 ? t('cron.empty.title') : t('cron.empty.noMatches')}
          />
        ) : (
          <div className="mx-auto w-full max-w-4xl min-h-0 flex-1 overflow-y-auto px-4 py-3">
            {/* Inline header replaces the old top-bar "New cron" button. We
                still need a single, always-visible affordance to add a job
                when the list is non-empty (rows themselves only expose
                edit/pause/trigger/delete). */}
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[0.7rem] uppercase tracking-wide text-muted-foreground">
                {t('cron.activeCount', { enabled: enabledCount, total: totalCount })}
              </span>
              <Button onClick={() => setEditor({ mode: 'create' })} size="sm">
                <Codicon name="add" />
                {t('cron.newCron')}
              </Button>
            </div>
            <div>
              {visibleJobs.map(job => (
                <CronJobRow
                  busy={busyJobId === job.id}
                  job={job}
                  key={job.id}
                  onDelete={() => setPendingDelete(job)}
                  onEdit={() => setEditor({ mode: 'edit', job })}
                  onPauseResume={() => void handlePauseResume(job)}
                  onTrigger={() => void handleTrigger(job)}
                  t={t}
                />
              ))}
            </div>
          </div>
        )}
      </div>
      <CronEditorDialog editor={editor} onClose={() => setEditor({ mode: 'closed' })} onSave={handleEditorSave} t={t} />

      <Dialog onOpenChange={open => !open && !deleting && setPendingDelete(null)} open={pendingDelete !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t('cron.deleteDialog.title')}</DialogTitle>
            <DialogDescription>
              {pendingDelete ? (
                <>
                  {t('cron.deleteDialog.before')}{' '}
                  <span className="font-medium text-foreground">{truncate(jobTitle(pendingDelete), 60)}</span>{' '}
                  {t('cron.deleteDialog.after')}
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button disabled={deleting} onClick={() => setPendingDelete(null)} variant="outline">
              {t('common.cancel')}
            </Button>
            <Button disabled={deleting} onClick={() => void handleConfirmDelete()} variant="destructive">
              {deleting ? t('cron.deleting') : t('cron.delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </OverlayView>
  )
}

function CronJobRow({
  busy,
  job,
  onDelete,
  onEdit,
  onPauseResume,
  onTrigger,
  t
}: {
  busy: boolean
  job: CronJob
  onDelete: () => void
  onEdit: () => void
  onPauseResume: () => void
  onTrigger: () => void
  t: Translate
}) {
  const state = jobState(job)
  const isPaused = state === 'paused'
  const hasName = Boolean(jobName(job))
  const prompt = jobPrompt(job)
  const deliver = jobDeliver(job)

  return (
    <div className="grid gap-3 px-3 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
      <button
        className="min-w-0 rounded-md text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        onClick={onEdit}
        type="button"
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="truncate text-sm font-medium">{jobTitle(job)}</span>
          <Badge className="capitalize" variant={STATE_VARIANT[state] ?? 'muted'}>
            {t(STATE_LABEL_KEYS[state] ?? 'cron.states.scheduled')}
          </Badge>
          {deliver && deliver !== DEFAULT_DELIVER && (
            <Badge className="capitalize" variant="muted">
              {deliver}
            </Badge>
          )}
        </div>
        {hasName && prompt && <p className="mt-1 truncate text-xs text-muted-foreground">{truncate(prompt, 120)}</p>}
        <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.68rem] text-muted-foreground">
          <span className="inline-flex items-center gap-1 font-mono">
            <Clock className="size-3" />
            {jobScheduleDisplay(job)}
          </span>
          <span>{t('cron.lastRun', { time: formatTime(job.last_run_at) })}</span>
          <span>{t('cron.nextRun', { time: formatTime(job.next_run_at) })}</span>
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
          <CronJobActionsTrigger
            className="text-muted-foreground hover:text-foreground"
            onClick={event => event.stopPropagation()}
            title={jobTitle(job)}
          />
        </CronJobActionsMenu>
      </div>
    </div>
  )
}

function EmptyState({
  actionLabel,
  description,
  onAction,
  title
}: {
  actionLabel?: string
  description: string
  onAction?: () => void
  title: string
}) {
  return (
    <div className="grid h-full place-items-center px-6 py-12 text-center">
      <div className="max-w-sm space-y-2">
        <div className="text-sm font-medium">{title}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
        {actionLabel && onAction && (
          <Button className="mt-2" onClick={onAction} size="sm">
            <Codicon name="add" />
            {actionLabel}
          </Button>
        )}
      </div>
    </div>
  )
}

function CronEditorDialog({
  editor,
  onClose,
  onSave,
  t
}: {
  editor: EditorState
  onClose: () => void
  onSave: (values: EditorValues) => Promise<void>
  t: Translate
}) {
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

  const scheduleHint = scheduleSummary(selectedScheduleOption, schedule, t)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const trimmedPrompt = prompt.trim()
    const trimmedSchedule = schedule.trim()

    if (!trimmedPrompt || !trimmedSchedule) {
      setError(t('cron.editor.requiredError'))

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
      setError(err instanceof Error ? err.message : t('cron.editor.saveFailed'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !saving && onClose()} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{isEdit ? t('cron.editor.editTitle') : t('cron.editor.newTitle')}</DialogTitle>
          <DialogDescription>
            {isEdit ? t('cron.editor.editDescription') : t('cron.editor.newDescription')}
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <Field htmlFor="cron-name" label={t('cron.editor.name')} optional optionalLabel={t('cron.editor.optional')}>
            <Input
              autoFocus
              id="cron-name"
              onChange={event => setName(event.target.value)}
              placeholder="Morning briefing"
              value={name}
            />
          </Field>

          <Field htmlFor="cron-prompt" label={t('cron.editor.prompt')}>
            <Textarea
              className="min-h-24 font-mono"
              id="cron-prompt"
              onChange={event => setPrompt(event.target.value)}
              placeholder="Summarize my unread Slack threads and email me the top 5..."
              value={prompt}
            />
          </Field>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="cron-frequency" label={t('cron.editor.frequency')}>
              <Select onValueChange={handleSchedulePresetChange} value={schedulePreset}>
                <SelectTrigger id="cron-frequency">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SCHEDULE_OPTIONS.map(option => (
                    <SelectItem key={option.value} value={option.value}>
                      {t(option.labelKey)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field htmlFor="cron-deliver" label={t('cron.editor.deliverTo')}>
              <Select onValueChange={setDeliver} value={deliver}>
                <SelectTrigger id="cron-deliver">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DELIVERY_OPTIONS.map(option => (
                    <SelectItem key={option.value} value={option.value}>
                      {t(option.labelKey)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          {schedulePreset === 'custom' ? (
            <Field htmlFor="cron-schedule" label={t('cron.editor.customSchedule')}>
              <Input
                className="font-mono"
                id="cron-schedule"
                onChange={event => setSchedule(event.target.value)}
                placeholder="0 9 * * * or weekdays at 9am"
                value={schedule}
              />
              <FieldHint>{t('cron.editor.customScheduleHint')}</FieldHint>
            </Field>
          ) : (
            <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-medium text-foreground">{scheduleHint}</span>
                <span className="font-mono text-muted-foreground">{schedule}</span>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={saving} onClick={onClose} type="button" variant="outline">
              {t('common.cancel')}
            </Button>
            <Button disabled={saving} type="submit">
              {saving ? t('common.saving') : isEdit ? t('cron.editor.saveChanges') : t('cron.editor.createCron')}
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
  optionalLabel,
  optional
}: {
  children: React.ReactNode
  htmlFor: string
  label: string
  optionalLabel?: string
  optional?: boolean
}) {
  return (
    <div className="grid gap-1.5">
      <label className="flex items-baseline gap-2 text-xs font-medium text-foreground" htmlFor={htmlFor}>
        {label}
        {optional && <span className="text-[0.65rem] font-normal text-muted-foreground">{optionalLabel}</span>}
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
  hintKey: string
  labelKey: string
  value: string
}
