import { createCronTriggerController, type CronTriggerController } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import {
  type AutomationBlueprint,
  createCronJob,
  type CronDeliveryTarget,
  type CronJob,
  deleteCronJob,
  getAutomationBlueprints,
  getCronDeliveryTargets,
  getCronJobRuns,
  instantiateAutomationBlueprint,
  pauseCronJob,
  resumeCronJob,
  type SessionInfo,
  updateCronJob
} from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { requestModelOptions } from '@/lib/model-options'
import { asText } from '@/lib/text'
import { $cronFocusJobId, $cronJobs, invalidateCronJobsRequests, setCronFocusJobId } from '@/store/cron'
import { $changeEventsAvailable, $cronChangeTick } from '@/store/live-sync'
import { notify, notifyError } from '@/store/notifications'
import { $profileScope, ALL_PROFILES } from '@/store/profile'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import {
  Panel,
  PanelAction,
  PanelAddButton,
  PanelBlock,
  PanelBody,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow,
  type PanelMenuItem,
  PanelMeta,
  PanelPill,
  type PanelPillTone,
  PanelSectionLabel
} from '../overlays/panel'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { BlueprintSlotControl, blueprintSlotHelp, cleanBlueprintFieldError, initialBlueprintValues } from './blueprints'
import { mutateAndRefreshCronJobs, refreshCronJobs, triggerAndRefreshCronJobs } from './cron-actions'
import {
  CRON_REASONING_VALUES,
  cronEditorUpdates,
  cronReasoningEffortToWire,
  type CronReasoningValue,
  jobIsScriptOnly,
  normalizeCronReasoningEffort,
  parseCronDeliveryTargets,
  toggleCronDeliveryTarget,
  validateCronEditor
} from './cron-job-model'
import { jobState, jobTitle, STATE_DOT } from './job-state'

const DEFAULT_DELIVER = 'local'

// Radix <SelectItem> rejects empty-string values, so the "no override" row in
// the model picker carries this sentinel and is mapped back to '' on save.
const MODEL_DEFAULT_VALUE = '__default__'

// "Start from" default: the manual editor (blank cron). Any other value is a
// blueprint key. Blueprint keys never collide with this sentinel.
const CUSTOM_TEMPLATE = 'custom'

function cronProfileForScope(scope: string): string {
  return scope === ALL_PROFILES ? 'all' : scope
}

const SCHEDULE_OPTIONS: ReadonlyArray<ScheduleOption> = [
  { expr: '0 9 * * *', value: 'daily' },
  { expr: '0 9 * * 1-5', value: 'weekdays' },
  { expr: '0 9 * * 1', value: 'weekly' },
  { expr: '0 9 1 * *', value: 'monthly' },
  { expr: '0 * * * *', value: 'hourly' },
  { expr: '*/15 * * * *', value: 'every-15-minutes' },
  { value: 'custom' }
]

const STATE_TONE: Record<string, PanelPillTone> = {
  enabled: 'good',
  scheduled: 'good',
  running: 'good',
  paused: 'warn',
  disabled: 'muted',
  error: 'bad',
  completed: 'muted'
}

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

function jobModel(job: CronJob): string {
  return asText(job.model).trim()
}

function jobProvider(job: CronJob): string {
  return asText(job.provider).trim()
}

function jobReasoningEffort(job: CronJob): CronReasoningValue {
  return normalizeCronReasoningEffort(job.reasoning_effort)
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
  onClose: () => void
  onOpenSession?: (sessionId: string) => void
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function CronView({ onClose, onOpenSession, setStatusbarItemGroup: _setStatusbarItemGroup }: CronViewProps) {
  const { t } = useI18n()
  const c = t.cron
  // Source of truth is the shared atom (also fed by the controller poll), so the
  // sidebar and this overlay never drift — a delete here clears the sidebar row
  // immediately. `loading` only gates the first paint before the atom is filled.
  const jobs = useStore($cronJobs)
  const [loading, setLoading] = useState(jobs.length === 0)
  const [query, setQuery] = useState('')
  const [busyJobTokens, setBusyJobTokens] = useState<ReadonlyMap<string, symbol>>(() => new Map())
  const [triggeringJobKeys, setTriggeringJobKeys] = useState<ReadonlySet<string>>(() => new Set())
  const triggerControllerRef = useRef<CronTriggerController | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- controller mount identity, not an atom mirror
  useEffect(() => {
    const controller = createCronTriggerController((key, running) => {
      if (triggerControllerRef.current !== controller) {
        return
      }

      setTriggeringJobKeys(current => {
        const next = new Set(current)

        if (running) {
          next.add(key)
        } else {
          next.delete(key)
        }

        return next
      })
    })

    triggerControllerRef.current = controller

    return () => {
      triggerControllerRef.current = null
    }
  }, [])

  // Master/detail: the job whose schedule + run history fill the right pane.
  const [selectedJobId, setSelectedJobId] = useState<null | string>(null)
  // Set when a job is opened from the sidebar so we scroll it into view once the
  // row exists. Cleared after the scroll fires.
  const pendingScrollRef = useRef<null | string>(null)
  const focusJobId = useStore($cronFocusJobId)

  const [editor, setEditor] = useState<EditorState>({ mode: 'closed' })
  const [pendingDelete, setPendingDelete] = useState<CronJob | null>(null)

  // Jobs live per-profile on disk and the list endpoint aggregates 'all' by
  // default — scope the fetch to the sidebar's profile scope so this overlay
  // and the sidebar (which share the $cronJobs atom) agree on what's shown.
  const profileScope = useStore($profileScope)
  const profile = cronProfileForScope(profileScope)

  const refresh = useCallback(async () => {
    const { refreshError, stale } = await refreshCronJobs(profile)

    if (stale) {
      return
    }

    if (refreshError) {
      notifyError(refreshError, c.failedLoad)
    }

    setLoading(false)
  }, [c, profile])

  useRefreshHotkey(refresh)

  useEffect(() => {
    void refresh()
    // Fence the previous profile's request before the next profile effect, and
    // fence every pending completion when the overlay unmounts.

    return () => invalidateCronJobsRequests()
  }, [refresh])

  // Sidebar → "open this job": resolve the focus id (or name) to a job, select
  // it, queue a scroll, then clear the one-shot focus so re-opening cron
  // normally doesn't re-trigger it.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!focusJobId) {
      return
    }

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

  // Blueprint recipes render in the same list rail, below the jobs — clicking
  // one opens the create dialog pre-seeded to that recipe. Same query key as
  // the dialog's "Start from" dropdown, so the catalog is fetched once.
  const blueprintsQuery = useQuery({
    queryKey: ['cron-blueprints'],
    queryFn: async () => (await getAutomationBlueprints()).blueprints
  })

  const visibleBlueprints = useMemo(() => {
    const list = blueprintsQuery.data ?? []
    const needle = query.trim().toLowerCase()

    return needle ? list.filter(item => `${item.title} ${item.description}`.toLowerCase().includes(needle)) : list
  }, [blueprintsQuery.data, query])

  // Detail always reflects a concrete job: the explicitly selected one, else the
  // first visible row, so the right pane is never empty while jobs exist.
  const selectedJob = useMemo(
    () => visibleJobs.find(job => job.id === selectedJobId) ?? visibleJobs[0] ?? null,
    [visibleJobs, selectedJobId]
  )

  // Scroll a sidebar-opened job into view once its list row is mounted.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const target = pendingScrollRef.current

    if (!target || selectedJob?.id !== target) {
      return
    }

    pendingScrollRef.current = null
    requestAnimationFrame(() => {
      document.querySelector(`[data-panel-row="${CSS.escape(target)}"]`)?.scrollIntoView({ block: 'nearest' })
    })
  }, [selectedJob])

  const totalCount = jobs.length

  function beginJobBusy(jobId: string): symbol {
    const token = Symbol(jobId)

    setBusyJobTokens(current => new Map(current).set(jobId, token))

    return token
  }

  function endJobBusy(jobId: string, token: symbol): void {
    setBusyJobTokens(current => {
      if (current.get(jobId) !== token) {
        return current
      }

      const next = new Map(current)

      next.delete(jobId)

      return next
    })
  }

  async function handlePauseResume(job: CronJob) {
    const busyToken = beginJobBusy(job.id)

    try {
      const isPaused = jobState(job) === 'paused'

      const { refreshError, stale } = await mutateAndRefreshCronJobs(profile, () =>
        isPaused ? resumeCronJob(job.id) : pauseCronJob(job.id)
      )

      if (stale) {
        return
      }

      if (refreshError) {
        notifyError(refreshError, c.failedLoad)
      }

      notify({
        kind: 'success',
        title: isPaused ? c.resumed : c.paused,
        message: truncate(jobTitle(job), 60)
      })
    } catch (err) {
      notifyError(err, c.failedUpdate)
    } finally {
      endJobBusy(job.id, busyToken)
    }
  }

  async function handleTrigger(job: CronJob) {
    const viewProfile = profile
    const key = `${viewProfile}:${job.id}`
    const controller = triggerControllerRef.current

    if (!controller) {
      return
    }

    try {
      const run = await controller.run(
        key,
        () => triggerAndRefreshCronJobs(job.id, viewProfile),
        () => notify({ kind: 'info', title: c.triggerNow, message: truncate(jobTitle(job), 60) })
      )

      if (
        triggerControllerRef.current !== controller ||
        cronProfileForScope($profileScope.get()) !== viewProfile ||
        !run.started ||
        !run.value
      ) {
        return
      }

      const { refreshError, stale } = run.value

      if (stale) {
        return
      }

      if (refreshError) {
        notifyError(refreshError, c.failedLoad)
      }

      notify({ kind: 'success', title: c.triggered, message: truncate(jobTitle(job), 60) })
    } catch (err) {
      if (triggerControllerRef.current === controller && cronProfileForScope($profileScope.get()) === viewProfile) {
        notifyError(err, c.failedTrigger)
      }
    }
  }

  // Throws on failure — ConfirmDialog reports it inline and stays open.
  async function handleConfirmDelete() {
    if (!pendingDelete) {
      return
    }

    const { refreshError, stale } = await mutateAndRefreshCronJobs(profile, () => deleteCronJob(pendingDelete.id))

    if (stale) {
      return
    }

    if (refreshError) {
      notifyError(refreshError, c.failedLoad)
    }

    notify({ kind: 'success', title: c.deleted, message: truncate(jobTitle(pendingDelete), 60) })
  }

  async function handleEditorSave(values: EditorValues) {
    if (editor.mode === 'create') {
      const {
        value: created,
        refreshError,
        stale
      } = await mutateAndRefreshCronJobs(profile, () =>
        createCronJob({
          prompt: values.prompt,
          schedule: values.schedule,
          name: values.name || undefined,
          deliver: values.deliver || DEFAULT_DELIVER,
          ...(values.model.trim() ? { model: values.model.trim(), provider: values.provider.trim() || undefined } : {}),
          reasoning_effort: cronReasoningEffortToWire(values.reasoningEffort)
        })
      )

      if (stale || !created) {
        return
      }

      if (refreshError) {
        notifyError(refreshError, c.failedLoad)
      }

      notify({ kind: 'success', title: c.created, message: truncate(jobTitle(created), 60) })
    } else if (editor.mode === 'edit') {
      const scriptOnlyJob = jobIsScriptOnly(editor.job)

      const {
        value: updated,
        refreshError,
        stale
      } = await mutateAndRefreshCronJobs(profile, () =>
        updateCronJob(editor.job.id, cronEditorUpdates(values, { scriptOnlyJob }))
      )

      if (stale || !updated) {
        return
      }

      if (refreshError) {
        notifyError(refreshError, c.failedLoad)
      }

      notify({ kind: 'success', title: c.updated, message: truncate(jobTitle(updated), 60) })
    }

    setEditor({ mode: 'closed' })
  }

  // Blueprint instantiation is a distinct backend path (fills typed slots, then
  // creates the job) so it can't share the raw-cron onSave contract. Merge the
  // created job into $cronJobs like every other create path. A blueprint writes a
  // real per-profile job, and "all" is not a writable target — collapse it to
  // 'default', matching the manual create path in handleEditorSave.
  async function handleBlueprintCreate(blueprint: AutomationBlueprint, values: Record<string, string>) {
    const writableProfile = profileScope === ALL_PROFILES ? 'default' : profileScope

    const {
      value: job,
      refreshError,
      stale
    } = await mutateAndRefreshCronJobs(profile, () =>
      instantiateAutomationBlueprint({ blueprint: blueprint.key, values }, writableProfile)
    )

    if (stale || !job) {
      return
    }

    if (refreshError) {
      notifyError(refreshError, c.failedLoad)
    }

    notify({ kind: 'success', title: c.blueprints.scheduled, message: asText(job.schedule_display) || blueprint.title })
    setEditor({ mode: 'closed' })
  }

  return (
    <Panel closeLabel={c.close} onClose={onClose}>
      <PanelHeader subtitle={c.count(totalCount)} title={c.title} />

      {loading && jobs.length === 0 ? (
        <PageLoader label={c.loading} />
      ) : totalCount === 0 && visibleBlueprints.length === 0 ? (
        <PanelEmpty
          action={
            <Button onClick={() => setEditor({ mode: 'create' })} size="sm">
              {c.newCron}
            </Button>
          }
          description={c.emptyDescNew}
          icon="watch"
          title={c.emptyTitleNew}
        />
      ) : (
        <PanelBody>
          <PanelList
            onSearchChange={setQuery}
            searchHints={jobs
              .map(jobTitle)
              .filter(Boolean)
              .slice(0, 5)
              .map(title => t.common.tryHint(title))}
            searchLabel={c.search}
            searchPlaceholder={c.search}
            searchValue={query}
          >
            {visibleJobs.map(job => (
              <CronJobListRow
                active={selectedJob?.id === job.id}
                job={job}
                key={job.id}
                menuItems={[
                  { icon: 'edit', label: c.edit, onSelect: () => setEditor({ mode: 'edit', job }) },
                  { icon: 'trash', label: t.common.delete, onSelect: () => setPendingDelete(job), tone: 'danger' }
                ]}
                menuLabel={c.manage}
                onSelect={() => setSelectedJobId(job.id)}
              />
            ))}
            {visibleJobs.length === 0 && (
              <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                {query.trim() ? c.emptyTitleSearch : c.emptyTitleNew}
              </p>
            )}
            <PanelAddButton label={c.newCron} onClick={() => setEditor({ mode: 'create' })} />
            {visibleBlueprints.length > 0 && (
              <>
                <PanelSectionLabel className="mt-3 px-2">{c.blueprints.tab}</PanelSectionLabel>
                {visibleBlueprints.map(item => (
                  <PanelListRow
                    active={false}
                    icon="rocket"
                    key={item.key}
                    onSelect={() => setEditor({ blueprintKey: item.key, mode: 'create' })}
                    rowKey={`blueprint-${item.key}`}
                    title={item.title}
                  />
                ))}
              </>
            )}
          </PanelList>

          {selectedJob ? (
            <CronJobDetail
              busy={busyJobTokens.has(selectedJob.id) || triggeringJobKeys.has(`${profile}:${selectedJob.id}`)}
              c={c}
              job={selectedJob}
              onOpenSession={onOpenSession}
              onPauseResume={() => void handlePauseResume(selectedJob)}
              onTrigger={() => void handleTrigger(selectedJob)}
            />
          ) : query.trim() ? (
            // A search with no selected job: search-flavored copy is right.
            <PanelEmpty description={c.emptyDescSearch} icon="search" />
          ) : (
            // No selection and no search — "Try a broader search query" here
            // just confused people staring at an empty panel with zero jobs.
            <PanelEmpty
              description={c.emptyDescNew}
              icon="watch"
              title={jobs.length === 0 ? c.emptyTitleNew : undefined}
            />
          )}
        </PanelBody>
      )}

      <CronEditorDialog
        editor={editor}
        onBlueprintCreate={handleBlueprintCreate}
        onClose={() => setEditor({ mode: 'closed' })}
        onSave={handleEditorSave}
      />

      <ConfirmDialog
        busyLabel={c.deleting}
        confirmLabel={t.common.delete}
        description={
          pendingDelete ? (
            <>
              {c.deleteDescPrefix}
              <span className="font-medium text-foreground">{truncate(jobTitle(pendingDelete), 60)}</span>
              {c.deleteDescSuffix}
            </>
          ) : null
        }
        destructive
        onClose={() => setPendingDelete(null)}
        onConfirm={handleConfirmDelete}
        open={pendingDelete !== null}
        title={c.deleteTitle}
      />
    </Panel>
  )
}

function CronJobListRow({
  active,
  job,
  menuItems,
  menuLabel,
  onSelect
}: {
  active: boolean
  job: CronJob
  menuItems?: PanelMenuItem[]
  menuLabel?: string
  onSelect: () => void
}) {
  const state = jobState(job)

  return (
    <PanelListRow
      active={active}
      dotClassName={STATE_DOT[state] ?? 'bg-muted-foreground'}
      menuItems={menuItems}
      menuLabel={menuLabel}
      onSelect={onSelect}
      rowKey={job.id}
      title={jobTitle(job)}
    />
  )
}

function CronJobDetail({
  busy,
  c,
  job,
  onOpenSession,
  onPauseResume,
  onTrigger
}: {
  busy: boolean
  c: Translations['cron']
  job: CronJob
  onOpenSession?: (sessionId: string) => void
  onPauseResume: () => void
  onTrigger: () => void
}) {
  const state = jobState(job)
  const isPaused = state === 'paused'
  const deliver = jobDeliver(job)
  const prompt = jobPrompt(job)
  const modelOverride = jobModel(job)
  const reasoningOverride = jobReasoningEffort(job)

  return (
    <PanelDetail>
      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{jobTitle(job)}</h3>
            <PanelPill tone={STATE_TONE[state] ?? 'muted'}>{c.states[state] ?? state}</PanelPill>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <PanelAction disabled={busy} icon={isPaused ? 'play' : 'debug-pause'} onClick={onPauseResume}>
              {isPaused ? c.resumeTitle : c.pauseTitle}
            </PanelAction>
            <PanelAction disabled={busy} icon="zap" onClick={onTrigger} primary>
              {c.triggerNow}
            </PanelAction>
          </div>
        </div>

        <PanelMeta
          rows={[
            { label: c.frequencyLabel, value: jobScheduleDisplay(job) },
            { label: c.last.replace(/:$/, ''), value: formatTime(job.last_run_at) },
            { label: c.next.replace(/:$/, ''), value: formatTime(job.next_run_at) },
            { label: c.deliverLabel, value: c.deliveryLabels[deliver] ?? deliver },
            ...(modelOverride ? [{ label: c.modelLabel, value: modelOverride }] : []),
            ...(reasoningOverride !== 'inherit'
              ? [{ label: c.reasoningLabel, value: c.reasoningLabels[reasoningOverride] ?? reasoningOverride }]
              : [])
          ]}
        />

        {job.last_error ? (
          <div className="flex items-start gap-1.5 rounded bg-destructive/10 p-2 text-[0.7rem] text-destructive">
            <AlertTriangle className="mt-px size-3 shrink-0" />
            <span className="min-w-0 break-words">{job.last_error}</span>
          </div>
        ) : null}
      </header>

      {prompt ? (
        <section className="space-y-1.5">
          <PanelSectionLabel>{c.promptLabel}</PanelSectionLabel>
          <PanelBlock>{prompt}</PanelBlock>
        </section>
      ) : null}

      <CronJobRuns c={c} jobId={job.id} onOpenSession={onOpenSession} />
    </PanelDetail>
  )
}

function formatRunTime(seconds?: null | number): string {
  if (!seconds) {
    return '—'
  }

  const date = new Date(seconds * 1000)

  return Number.isNaN(date.valueOf()) ? '—' : date.toLocaleString()
}

// Runs are produced by the background scheduler tick. cron.changed /
// sessions.changed broadcasts re-load immediately on event-capable backends
// (the tick dep below), so the poll drops to a slow backstop there; older
// backends keep the legacy cadence.
const RUNS_POLL_INTERVAL_MS = 8000
const RUNS_BACKSTOP_INTERVAL_MS = 60_000

function CronJobRuns({
  c,
  jobId,
  onOpenSession
}: {
  c: Translations['cron']
  jobId: string
  onOpenSession?: (sessionId: string) => void
}) {
  const [runs, setRuns] = useState<null | SessionInfo[]>(null)
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)

  useEffect(() => {
    let cancelled = false

    const load = () =>
      getCronJobRuns(jobId)
        .then(result => {
          if (!cancelled) {
            setRuns(result)
          }
        })
        .catch(() => {
          if (!cancelled) {
            setRuns(prev => prev ?? [])
          }
        })

    void load()

    const intervalId = window.setInterval(
      () => {
        if (document.visibilityState === 'visible') {
          void load()
        }
      },
      changeEventsAvailable ? RUNS_BACKSTOP_INTERVAL_MS : RUNS_POLL_INTERVAL_MS
    )

    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        void load()
      }
    }

    document.addEventListener('visibilitychange', onVisible)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
      document.removeEventListener('visibilitychange', onVisible)
    }
    // cronChangeTick: a fired run moves jobs.json bookkeeping → reload now.
  }, [changeEventsAvailable, cronChangeTick, jobId])

  return (
    <div>
      <PanelSectionLabel className="mb-1.5">
        {c.runHistory}
        {runs && runs.length > 0 ? ` · ${runs.length}` : ''}
      </PanelSectionLabel>
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
              className="row-hover flex items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              key={run.id}
              onClick={() => onOpenSession?.(run.id)}
              type="button"
            >
              <span className="truncate text-foreground/85">{run.title?.trim() || run.preview?.trim() || run.id}</span>
              <span className="shrink-0 text-[0.62rem] text-muted-foreground/55 tabular-nums">
                {formatRunTime(run.last_active || run.started_at)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// Label a cron delivery target: 'local' → localized "This desktop", known
// platforms → their delivery label, anything else → the backend name. Configured
// platforms without a cron home channel get a "set a home channel first" hint.
function deliverTargetLabel(target: CronDeliveryTarget, c: Translations['cron']): string {
  const base = target.id === 'local' ? c.deliveryLabels.local : (c.deliveryLabels[target.id] ?? target.name)

  return target.id !== 'local' && !target.home_target_set ? `${base} — ${c.deliverNeedsHomeChannel}` : base
}

// The delivery-target checkbox group, shared by the manual cron editor and the
// blueprint form. The scheduler accepts comma-separated targets, so users can
// keep results local while also sending them to connected platforms. Preserve
// selected targets missing from discovery so editing never drops a saved route.
export function DeliverCheckboxes({
  c,
  id,
  onChange,
  targets,
  value
}: {
  c: Translations['cron']
  id: string
  onChange: (next: string) => void
  targets: CronDeliveryTarget[]
  value: string
}) {
  const selected = parseCronDeliveryTargets(value)
  const knownIds = new Set(targets.map(target => target.id))

  const options = [
    ...targets,
    ...selected
      .filter(target => !knownIds.has(target))
      .map(target => ({ home_env_var: null, home_target_set: true, id: target, name: target }))
  ]

  return (
    <div
      aria-labelledby={`${id}-label`}
      className="grid gap-2 rounded-md border border-input px-3 py-2.5"
      id={id}
      role="group"
    >
      {options.map((target, index) => {
        const checked = selected.includes(target.id)
        const checkboxId = `${id}-${index}`

        return (
          <label className="flex items-center gap-2 text-sm" htmlFor={checkboxId} key={target.id}>
            <Checkbox
              checked={checked}
              id={checkboxId}
              onCheckedChange={next => onChange(toggleCronDeliveryTarget(value, target.id, next === true))}
            />
            <span>{deliverTargetLabel(target, c)}</span>
          </label>
        )
      })}
    </div>
  )
}

function CronEditorDialog({
  editor,
  onBlueprintCreate,
  onClose,
  onSave
}: {
  editor: EditorState
  onBlueprintCreate: (blueprint: AutomationBlueprint, values: Record<string, string>) => Promise<void>
  onClose: () => void
  onSave: (values: EditorValues) => Promise<void>
}) {
  const { t } = useI18n()
  const c = t.cron
  const open = editor.mode !== 'closed'
  const isEdit = editor.mode === 'edit'
  const initial = isEdit ? editor.job : null
  const scriptOnlyJob = initial ? jobIsScriptOnly(initial) : false

  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')
  const [schedule, setSchedule] = useState('')
  const [schedulePreset, setSchedulePreset] = useState('daily')
  const [deliver, setDeliver] = useState(DEFAULT_DELIVER)
  // Per-job model override, encoded as `${providerSlug}:${model}` (split on the
  // first ':' when saving). MODEL_DEFAULT_VALUE = follow the global default.
  const [modelChoice, setModelChoice] = useState(MODEL_DEFAULT_VALUE)
  // Per-job reasoning effort. 'inherit' = follow config resolution at fire
  // time; see CRON_REASONING_VALUES in cron-job-model.ts.
  const [reasoningEffort, setReasoningEffort] = useState<CronReasoningValue>('inherit')
  // Blueprint fills typed slots (time/enum/weekdays/text) instead of the raw
  // cron fields; the backend renders the prompt + schedule from them.
  const [slotValues, setSlotValues] = useState<Record<string, string>>({})
  // Create mode can start from a ready-made blueprint instead of a blank cron.
  // CUSTOM_TEMPLATE (default) = the manual editor; any other value is a
  // blueprint key that swaps the form for that blueprint's typed slots.
  const [templateChoice, setTemplateChoice] = useState(CUSTOM_TEMPLATE)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // The blueprint catalog powers the create dialog's "Start from" dropdown; it's
  // meaningless when editing an existing job, so skip the fetch there.
  const blueprintsQuery = useQuery({
    queryKey: ['cron-blueprints'],
    queryFn: async () => (await getAutomationBlueprints()).blueprints,
    enabled: open && !isEdit
  })

  const blueprintList = blueprintsQuery.data ?? []

  const blueprint =
    templateChoice === CUSTOM_TEMPLATE ? null : (blueprintList.find(item => item.key === templateChoice) ?? null)

  const isBlueprint = blueprint !== null

  // Same catalog the chat model picker uses: configured providers and their
  // actually-available models only. Script-only + blueprint forms never pick a
  // model here, so skip the fetch entirely for them.
  const modelOptions = useQuery({
    queryKey: ['model-options', 'global'],
    queryFn: () => requestModelOptions({}),
    enabled: open && !scriptOnlyJob && !isBlueprint
  })

  // Single source of truth for where a cron can deliver (local + configured
  // gateways) — same endpoint the dashboard uses, so no dialog offers a platform
  // that isn't connected. Shared by the manual editor and the blueprint form.
  const deliveryTargets = useQuery({
    queryKey: ['cron-delivery-targets'],
    queryFn: getCronDeliveryTargets,
    enabled: open
  })

  useEffect(() => {
    if (!open) {
      return
    }

    setName(initial ? jobName(initial) : '')
    setPrompt(initial ? jobPrompt(initial) : '')
    setSchedule(initial ? jobScheduleExpr(initial) : (SCHEDULE_OPTIONS[0].expr ?? ''))
    setSchedulePreset(initial ? scheduleOptionForExpr(jobScheduleExpr(initial)).value : 'daily')
    setDeliver(initial ? jobDeliver(initial) : DEFAULT_DELIVER)
    setModelChoice(initial && jobModel(initial) ? `${jobProvider(initial)}:${jobModel(initial)}` : MODEL_DEFAULT_VALUE)
    setReasoningEffort(initial ? jobReasoningEffort(initial) : 'inherit')
    setSlotValues({})
    setTemplateChoice(editor.mode === 'create' ? (editor.blueprintKey ?? CUSTOM_TEMPLATE) : CUSTOM_TEMPLATE)
    setError(null)
    setSaving(false)
  }, [editor, initial, open])

  // Seed the typed slots with the blueprint's defaults whenever a blueprint is
  // picked from "Start from" (and reset them when switching back to Custom).
  useEffect(() => {
    setSlotValues(blueprint ? initialBlueprintValues(blueprint) : {})
    setError(null)
  }, [blueprint])

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

  // Configured providers with at least one available model — mirrors the chat
  // model picker's gate so only actually-selectable models are offered.
  const modelProviders = (modelOptions.data?.providers ?? []).filter(
    provider => provider.authenticated !== false && (provider.models ?? []).length > 0
  )

  // A previously pinned model that has since left the catalog (provider
  // removed / model retired) would render Radix's blank trigger. Keep the
  // stored pin visible and re-selectable rather than silently dropping it.
  const modelChoiceKnown =
    modelChoice === MODEL_DEFAULT_VALUE ||
    modelProviders.some(provider => (provider.models ?? []).some(model => `${provider.slug}:${model}` === modelChoice))

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    const validationError = validateCronEditor({
      prompt,
      schedule,
      scriptOnlyJob
    })

    if (validationError) {
      setError(
        validationError === 'schedule'
          ? c.scheduleRequired
          : validationError === 'prompt'
            ? c.promptRequired
            : c.promptScheduleRequired
      )

      return
    }

    // Decode `${providerSlug}:${model}` — the model half may itself contain
    // ':' (e.g. openrouter 'anthropic/claude-sonnet-4:beta'), so split once.
    const overrideIndex = modelChoice === MODEL_DEFAULT_VALUE ? -1 : modelChoice.indexOf(':')
    const overrideProvider = overrideIndex >= 0 ? modelChoice.slice(0, overrideIndex) : ''
    const overrideModel = overrideIndex >= 0 ? modelChoice.slice(overrideIndex + 1) : ''

    setSaving(true)
    setError(null)

    try {
      await onSave({
        deliver,
        model: overrideModel,
        name: name.trim(),
        prompt: prompt.trim(),
        provider: overrideProvider,
        reasoningEffort,
        schedule: schedule.trim()
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : c.failedSave)
    } finally {
      setSaving(false)
    }
  }

  async function handleBlueprintSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!blueprint) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      await onBlueprintCreate(blueprint, slotValues)
    } catch (err) {
      // 422 carries the slot-level validation message; surface it inline.
      setError(cleanBlueprintFieldError(err instanceof Error ? err.message : String(err)))
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

        {!isEdit && blueprintList.length > 0 && (
          <Field htmlFor="cron-template" label={c.blueprints.startFrom}>
            <Select onValueChange={setTemplateChoice} value={templateChoice}>
              <SelectTrigger className="h-9 rounded-md" id="cron-template">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={CUSTOM_TEMPLATE}>{c.blueprints.custom}</SelectItem>
                {blueprintList.map(item => (
                  <SelectItem key={item.key} value={item.key}>
                    {item.title}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {blueprint?.description && <FieldHint>{blueprint.description}</FieldHint>}
          </Field>
        )}

        {isBlueprint && blueprint ? (
          <form className="grid gap-4" onSubmit={handleBlueprintSubmit}>
            {blueprint.fields.map(field => {
              const fieldId = `blueprint-${blueprint.key}-${field.name}`
              const help = blueprintSlotHelp(field)

              return (
                <Field htmlFor={fieldId} key={field.name} label={field.label}>
                  {field.name === 'deliver' ? (
                    // Use the shared, backend-sourced delivery targets (same as the
                    // manual editor) rather than the blueprint's static field.options,
                    // so both dialogs offer exactly the connected platforms.
                    <DeliverCheckboxes
                      c={c}
                      id={fieldId}
                      onChange={next => setSlotValues(prev => ({ ...prev, [field.name]: next }))}
                      targets={deliveryTargets.data ?? []}
                      value={slotValues[field.name] ?? DEFAULT_DELIVER}
                    />
                  ) : (
                    <BlueprintSlotControl
                      field={field}
                      id={fieldId}
                      onChange={next => setSlotValues(prev => ({ ...prev, [field.name]: next }))}
                      value={slotValues[field.name] ?? ''}
                    />
                  )}
                  {help && <FieldHint>{help}</FieldHint>}
                </Field>
              )
            })}

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
                {saving ? c.blueprints.scheduling : c.blueprints.scheduleIt}
              </Button>
            </DialogFooter>
          </form>
        ) : (
          <form className="grid gap-4" onSubmit={handleSubmit}>
            {scriptOnlyJob && initial && (
              <FieldHint>
                {c.scriptOnlyEditHint} <span className="font-mono">{initial.id}</span>
              </FieldHint>
            )}

            <Field htmlFor="cron-name" label={c.nameLabel} optional optionalLabel={c.optional}>
              <Input
                autoFocus
                id="cron-name"
                onChange={event => setName(event.target.value)}
                placeholder={c.namePlaceholder}
                value={name}
              />
            </Field>

            <Field htmlFor="cron-prompt" label={c.promptLabel} optional={scriptOnlyJob} optionalLabel={c.optional}>
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
                <DeliverCheckboxes
                  c={c}
                  id="cron-deliver"
                  onChange={setDeliver}
                  targets={deliveryTargets.data ?? []}
                  value={deliver}
                />
              </Field>
            </div>

            {!scriptOnlyJob && (
              <Field htmlFor="cron-model" label={c.modelLabel} optional optionalLabel={c.optional}>
                <Select onValueChange={setModelChoice} value={modelChoice}>
                  <SelectTrigger className="h-9 rounded-md" id="cron-model">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={MODEL_DEFAULT_VALUE}>{c.modelDefault}</SelectItem>
                    {!modelChoiceKnown && (
                      <SelectItem className="font-mono" value={modelChoice}>
                        {modelChoice.slice(modelChoice.indexOf(':') + 1)}
                      </SelectItem>
                    )}
                    {modelProviders.map(provider => (
                      <SelectGroup key={provider.slug}>
                        <SelectLabel>{provider.name}</SelectLabel>
                        {(provider.models ?? []).map(model => (
                          <SelectItem
                            className="font-mono"
                            key={`${provider.slug}:${model}`}
                            value={`${provider.slug}:${model}`}
                          >
                            {model}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}

            {!scriptOnlyJob && (
              <Field htmlFor="cron-reasoning" label={c.reasoningLabel} optional optionalLabel={c.optional}>
                <Select
                  onValueChange={value => setReasoningEffort(normalizeCronReasoningEffort(value))}
                  value={reasoningEffort}
                >
                  <SelectTrigger className="h-9 rounded-md" id="cron-reasoning">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CRON_REASONING_VALUES.map(value => (
                      <SelectItem key={value} value={value}>
                        {c.reasoningLabels[value] ?? value}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}

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
        )}
      </DialogContent>
    </Dialog>
  )
}

type EditorState =
  | { job: CronJob; mode: 'edit' }
  | { mode: 'closed' }
  // `blueprintKey` pre-selects a blueprint in the create dialog's "Start from"
  // dropdown (set when a recipe row in the list rail is clicked).
  | { blueprintKey?: string; mode: 'create' }

interface EditorValues {
  deliver: string
  /** Per-job model override ('' = follow the global default). */
  model: string
  name: string
  prompt: string
  /** Provider slug for the model override ('' = none). */
  provider: string
  /** Per-job reasoning effort ('inherit' = follow config resolution). */
  reasoningEffort: CronReasoningValue
  schedule: string
}

interface ScheduleOption {
  expr?: string
  value: string
}
