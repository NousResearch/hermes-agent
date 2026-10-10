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
import { Tip } from '@/components/ui/tooltip'
import {
  type AutomationBlueprint,
  type AutomationBlueprintJobSpec,
  createCronJob,
  type CronDeliveryTarget,
  type CronJob,
  deleteCronJob,
  getAutomationBlueprints,
  getCronDeliveryTargets,
  getCronJobRuns,
  instantiateAutomationBlueprint,
  pauseCronJob,
  renderAutomationBlueprint,
  resumeCronJob,
  type SessionInfo,
  updateCronJob
} from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { requestModelOptions } from '@/lib/model-options'
import { asText } from '@/lib/text'
import {
  $cronFocusJobId,
  $cronJobs,
  type CronEditorDraft,
  type CronEditorDraftValues,
  invalidateCronJobsRequests,
  parkCronEditorDraft,
  setCronFocusJobId,
  takeCronEditorDraft
} from '@/store/cron'
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
  cronEditorUpdates,
  cronModelChoiceValue,
  jobDescription,
  jobIsScriptOnly,
  lastErrorSummary,
  parseCronDeliveryTargets,
  parseCronModelChoiceValue,
  toggleCronDeliveryTarget,
  validateCronEditor
} from './cron-job-model'
import { jobState, jobTitle, nextRunOverdueMs, STATE_DOT, truncateText } from './job-state'
import { openCronRun, reconcileCronRunVerdicts } from './open-cron-run'
import { SCHEDULE_OPTIONS, scheduleOptionForExpr } from './schedule'
import { ScheduleFields } from './schedule-fields'
import { BLANK_START, carriedFields, copyableJobs, StartFromField, startJob } from './start-from'

const DEFAULT_DELIVER = 'local'

// Radix <SelectItem> rejects empty-string values, so the "no override" row in
// the model picker carries this sentinel and is mapped back to '' on save.
const MODEL_DEFAULT_VALUE = '__default__'

function cronProfileForScope(scope: string): string {
  return scope === ALL_PROFILES ? 'all' : scope
}

// A blueprint writes a real per-profile job, and "all" is not a writable target —
// collapse it to 'default', matching the manual create path in handleEditorSave.
// The catalog is fetched for the same profile: plugin blueprints are per profile.
function blueprintProfileForScope(scope: string): string {
  return scope === ALL_PROFILES ? 'default' : scope
}

const STATE_TONE: Record<string, PanelPillTone> = {
  enabled: 'good',
  scheduled: 'good',
  running: 'good',
  paused: 'warn',
  disabled: 'muted',
  error: 'bad',
  completed: 'muted'
}

const truncate = (value: string, max = 80): string => truncateText(value, max)

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
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
  /** Leave for Messaging settings (connect a delivery platform); the editor draft is parked first. */
  onOpenMessaging?: () => void
  /** Open a fresh chat with this prompt typed in, unsent; the editor draft is parked first. */
  onTestPrompt?: (prompt: string) => void
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function CronView({
  onClose,
  onOpenMessaging,
  onOpenSession,
  onTestPrompt,
  setStatusbarItemGroup: _setStatusbarItemGroup
}: CronViewProps) {
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
  // Accepted-but-unmaterialized triggers, by `${profile}:${jobId}`. The ref is
  // written before the state set, so a queued row appears synchronously with
  // the click and the controller guard has no untracked window.
  const pendingTriggersRef = useRef(new Map<string, number>())
  const [pendingTriggers, setPendingTriggers] = useState<ReadonlyMap<string, number>>(() => new Map())

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

  // A draft parked by "Test in new chat" / "Connect a platform" reopens the
  // editor once. An edit waits for its job to load; a job deleted meanwhile
  // drops the draft rather than resurrecting it as a new job.
  const [parkedDraft, setParkedDraft] = useState<CronEditorDraft | null>(null)

  useEffect(() => {
    const draft = takeCronEditorDraft(profile)

    if (draft) {
      setParkedDraft(draft)
    }
  }, [profile])

  useEffect(() => {
    if (!parkedDraft) {
      return
    }

    const job = jobs.find(row => row.id === parkedDraft.jobId)

    if (parkedDraft.jobId !== null && !job && loading) {
      return
    }

    if (parkedDraft.jobId === null) {
      setEditor({ draft: parkedDraft.values, mode: 'create' })
    } else if (job) {
      setEditor({ draft: parkedDraft.values, job, mode: 'edit' })
    }

    setParkedDraft(null)
  }, [jobs, loading, parkedDraft])

  function parkEditorDraft(values: CronEditorDraftValues) {
    parkCronEditorDraft({ jobId: editor.mode === 'edit' ? editor.job.id : null, profile, values })
  }

  const handleTestPrompt = onTestPrompt
    ? (values: CronEditorDraftValues) => {
        parkEditorDraft(values)
        onTestPrompt(values.prompt)
      }
    : undefined

  const handleOpenMessaging = onOpenMessaging
    ? (values: CronEditorDraftValues) => {
        parkEditorDraft(values)
        onOpenMessaging()
      }
    : undefined

  const visibleJobs = useMemo(
    () => jobs.filter(job => matchesQuery(job, query.trim())).sort((a, b) => jobTitle(a).localeCompare(jobTitle(b))),
    [jobs, query]
  )

  // Blueprint recipes render in the same list rail, below the jobs — clicking
  // one opens the create dialog pre-seeded to that recipe. Same query key as
  // the dialog's "Start from" dropdown, so the catalog is fetched once.
  const blueprintProfile = blueprintProfileForScope(profileScope)

  const blueprintsQuery = useQuery({
    queryKey: ['cron-blueprints', blueprintProfile],
    queryFn: async () => (await getAutomationBlueprints(blueprintProfile)).blueprints
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

  const settlePendingTrigger = useCallback((key: string, requestedAt: number) => {
    if (pendingTriggersRef.current.get(key) !== requestedAt) {
      return
    }

    pendingTriggersRef.current.delete(key)
    setPendingTriggers(current => {
      if (current.get(key) !== requestedAt) {
        return current
      }

      const next = new Map(current)

      next.delete(key)

      return next
    })
  }, [])

  async function handleTrigger(job: CronJob) {
    const viewProfile = profile
    const key = `${viewProfile}:${job.id}`
    const controller = triggerControllerRef.current

    if (!controller) {
      return
    }

    const requestedAt = Date.now()

    // Optimistic queued-run feedback: the row is painted from the click, not
    // from the backend materializing the session (which can take tens of
    // seconds); it settles when Run History observes the run or times out.
    // Ref first so the busy render and the controller guard share one instant.
    pendingTriggersRef.current.set(key, requestedAt)
    setPendingTriggers(current => new Map(current).set(key, requestedAt))

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

      // The request never reached the backend; the queued row is a lie.
      settlePendingTrigger(key, requestedAt)
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
          ...values.carried,
          prompt: values.prompt,
          schedule: values.schedule,
          name: values.name || undefined,
          deliver: values.deliver || DEFAULT_DELIVER,
          ...(values.model.trim() ? { model: values.model.trim(), provider: values.provider.trim() || undefined } : {})
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
  // created job into $cronJobs like every other create path.
  async function handleBlueprintCreate(blueprint: AutomationBlueprint, values: Record<string, string>) {
    const writableProfile = blueprintProfile

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
                    meta={item.plugin || undefined}
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
              onEdit={() => setEditor({ mode: 'edit', job: selectedJob })}
              onOpenSession={onOpenSession}
              onPauseResume={() => void handlePauseResume(selectedJob)}
              onPendingRunSettled={settlePendingTrigger}
              onTrigger={() => void handleTrigger(selectedJob)}
              pendingJobKey={`${profile}:${selectedJob.id}`}
              pendingRunAt={pendingTriggers.get(`${profile}:${selectedJob.id}`)}
            />
          ) : query.trim() ? (
            // A search with no selected job: search-flavored copy is right.
            <PanelEmpty description={c.emptyDescSearch} icon="search" />
          ) : (
            // No selection and no search — "Try a broader search query" here
            // just confused people staring at an empty panel with zero jobs.
            <PanelEmpty
              action={
                jobs.length === 0 ? (
                  <Button onClick={() => setEditor({ mode: 'create' })} size="sm">
                    {c.newCron}
                  </Button>
                ) : undefined
              }
              description={c.emptyDescNew}
              icon="watch"
              title={jobs.length === 0 ? c.emptyTitleNew : undefined}
            />
          )}
        </PanelBody>
      )}

      <CronEditorDialog
        blueprintProfile={blueprintProfile}
        editor={editor}
        jobs={jobs}
        onBlueprintCreate={handleBlueprintCreate}
        onClose={() => setEditor({ mode: 'closed' })}
        onOpenMessaging={handleOpenMessaging}
        onSave={handleEditorSave}
        onTestPrompt={handleTestPrompt}
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

interface CronJobDetailProps {
  busy: boolean
  c: Translations['cron']
  job: CronJob
  onEdit: () => void
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
  onPauseResume: () => void
  onPendingRunSettled: (key: string, requestedAt: number) => void
  onTrigger: () => void
  pendingJobKey?: string
  pendingRunAt?: number
}

function CronJobDetail({
  busy,
  c,
  job,
  onEdit,
  onOpenSession,
  onPauseResume,
  onPendingRunSettled,
  onTrigger,
  pendingJobKey,
  pendingRunAt
}: CronJobDetailProps) {
  const state = jobState(job)
  const isPaused = state === 'paused'
  const deliver = jobDeliver(job)
  const prompt = jobPrompt(job)
  const scriptOnly = jobIsScriptOnly(job)
  const description = jobDescription(job)
  const modelOverride = jobModel(job)

  return (
    <PanelDetail>
      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <h3 className="text-[0.95rem] font-semibold tracking-tight text-foreground">{jobTitle(job)}</h3>
            {scriptOnly && <PanelPill tone="muted">{c.scriptBadge}</PanelPill>}
            <PanelPill tone={STATE_TONE[state] ?? 'muted'}>{c.states[state] ?? state}</PanelPill>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <PanelAction disabled={busy} icon={isPaused ? 'play' : 'debug-pause'} onClick={onPauseResume}>
              {isPaused ? c.resumeTitle : c.pauseTitle}
            </PanelAction>
            <PanelAction
              disabled={busy}
              icon={pendingRunAt === undefined ? 'zap' : 'loading'}
              onClick={onTrigger}
              primary
              spinning={pendingRunAt !== undefined}
            >
              {c.triggerNow}
            </PanelAction>
          </div>
        </div>

        <PanelMeta
          rows={[
            { label: c.frequencyLabel, value: jobScheduleDisplay(job) },
            { label: c.last.replace(/:$/, ''), value: formatTime(job.last_run_at) },
            {
              label: (nextRunOverdueMs(job) === null ? c.next : c.overdueSince).replace(/:$/, ''),
              value: formatTime(job.next_run_at)
            },
            { label: c.deliverLabel, value: c.deliveryLabels[deliver] ?? deliver },
            ...(modelOverride ? [{ label: c.modelLabel, value: modelOverride }] : [])
          ]}
        />

        {job.last_error ? (
          <div className="space-y-1.5 rounded bg-destructive/10 p-2 text-[0.7rem] text-destructive">
            <div className="flex items-start gap-1.5">
              <AlertTriangle className="mt-px size-3 shrink-0" />
              <span className="min-w-0 break-words" title={job.last_error}>
                {c.lastRunFailed} {lastErrorSummary(job.last_error)}
              </span>
            </div>
            <div className="flex items-center gap-0.5 pl-4">
              <PanelAction disabled={busy} icon="edit" onClick={onEdit}>
                {c.editJob}
              </PanelAction>
              <PanelAction disabled={busy} icon="zap" onClick={onTrigger}>
                {c.runAgain}
              </PanelAction>
            </div>
          </div>
        ) : null}
      </header>

      {description ? (
        <section className="space-y-1.5">
          <PanelSectionLabel>{scriptOnly && !prompt ? c.scriptLabel : c.promptLabel}</PanelSectionLabel>
          <PanelBlock>{description}</PanelBlock>
        </section>
      ) : null}

      <CronJobRuns
        c={c}
        jobId={job.id}
        onOpenSession={onOpenSession}
        onPendingRunSettled={onPendingRunSettled}
        pendingJobKey={pendingJobKey}
        pendingRunAt={pendingRunAt}
      />
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

// Script-only (no_agent) jobs have no agent sessions; the runs endpoint
// surfaces their per-fire output docs as rows with source='cron_output'
// (see _list_cron_output_runs in hermes_cli/web_routers/cron.py).
function isSyntheticCronOutputRun(run: SessionInfo): boolean {
  return run.source === 'cron_output'
}

// Runs are produced by the background scheduler tick. cron.changed /
// sessions.changed broadcasts re-load immediately on event-capable backends
// (the tick dep below), so the poll drops to a slow backstop there; older
// backends keep the legacy cadence. While a trigger is queued, poll fast.
const RUNS_POLL_INTERVAL_MS = 8000
const RUNS_BACKSTOP_INTERVAL_MS = 60_000
const PENDING_RUNS_POLL_INTERVAL_MS = 1000
// A run created moments before the click (another surface's trigger, or a
// scheduler tick racing the button) must not be mistaken for this click's run
// — but clock skew between renderer and backend can date it slightly early.
const PENDING_RUN_START_SLACK_MS = 2000
// Bounded even if the backend never materializes the run (claimed by a window
// that died, scheduler paused, …): the queued row is transient feedback, not a
// persistent record.
const PENDING_RUN_TIMEOUT_MS = 90_000

function CronJobRuns({
  c,
  jobId,
  onOpenSession,
  onPendingRunSettled,
  pendingJobKey,
  pendingRunAt
}: {
  c: Translations['cron']
  jobId: string
  onOpenSession?: (sessionId: string, session?: SessionInfo) => void
  onPendingRunSettled: (key: string, requestedAt: number) => void
  pendingJobKey?: string
  pendingRunAt?: number
}) {
  const [runs, setRuns] = useState<null | SessionInfo[]>(null)
  const changeEventsAvailable = useStore($changeEventsAvailable)
  const cronChangeTick = useStore($cronChangeTick)

  const pending = pendingRunAt !== undefined

  useEffect(() => {
    let cancelled = false

    const load = () =>
      getCronJobRuns(jobId)
        .then(result => {
          // A fresh poll re-evaluates every run already opened (#88443).
          reconcileCronRunVerdicts(result)

          if (!cancelled) {
            setRuns(result)

            // The queued row is settled by the run this trigger produced:
            // any run that started after the click. Looking at the whole
            // snapshot (not just a diff against the last poll) avoids
            // missing a run that landed between two loads.
            if (pendingRunAt !== undefined && pendingJobKey !== undefined) {
              const started = result.some(run => {
                const startedAtMs = (run.started_at || run.last_active || 0) * 1000

                return startedAtMs >= pendingRunAt - PENDING_RUN_START_SLACK_MS
              })

              if (started) {
                onPendingRunSettled(pendingJobKey, pendingRunAt)
              }
            }
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
      pending
        ? PENDING_RUNS_POLL_INTERVAL_MS
        : changeEventsAvailable
          ? RUNS_BACKSTOP_INTERVAL_MS
          : RUNS_POLL_INTERVAL_MS
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
  }, [changeEventsAvailable, cronChangeTick, jobId, onPendingRunSettled, pending, pendingJobKey, pendingRunAt])

  // Bounded life for the queued row: even on an event-capable backend (where
  // the fast poll above may be the only prober), the feedback disappears and
  // the action unlocks once the wait is clearly unrecoverable.
  useEffect(() => {
    if (pendingRunAt === undefined || pendingJobKey === undefined) {
      return
    }

    const remainingMs = Math.max(0, pendingRunAt + PENDING_RUN_TIMEOUT_MS - Date.now())
    const timeoutId = window.setTimeout(() => onPendingRunSettled(pendingJobKey, pendingRunAt), remainingMs)

    return () => window.clearTimeout(timeoutId)
  }, [onPendingRunSettled, pendingJobKey, pendingRunAt])

  return (
    <div>
      <PanelSectionLabel className="mb-1.5">
        {c.runHistory}
        {runs && runs.length + (pending ? 1 : 0) > 0 ? ` · ${runs.length + (pending ? 1 : 0)}` : ''}
      </PanelSectionLabel>
      {runs === null && !pending ? (
        <div className="flex items-center gap-1.5 py-1 text-xs text-muted-foreground">
          <Codicon name="loading" size="0.75rem" spinning />
        </div>
      ) : runs?.length === 0 && !pending ? (
        <div className="py-1 text-xs text-muted-foreground">{c.noRuns}</div>
      ) : (
        <div className="flex flex-col gap-px">
          {pending && (
            // The queued row is transient feedback for a trigger the backend
            // accepted but has not turned into a session yet; it sits above
            // the authoritative rows and is replaced once one appears.
            <div
              aria-live="polite"
              className="flex items-center justify-between gap-3 rounded-md px-2 py-1 text-xs text-muted-foreground"
              data-slot="cron-run-pending"
              role="status"
            >
              <span className="flex min-w-0 items-center gap-1.5">
                <Codicon name="loading" size="0.75rem" spinning />
                <span className="truncate">{c.queuedRun}</span>
              </span>
              <span className="shrink-0 text-[0.62rem] text-muted-foreground/55 tabular-nums">
                {formatRunTime(pendingRunAt / 1000)}
              </span>
            </div>
          )}
          {(runs ?? []).map(run =>
            isSyntheticCronOutputRun(run) ? (
              // Output-doc rows have no backing session to open; show the
              // recorded output preview without a chat-navigation affordance.
              <div className="flex items-center justify-between gap-3 rounded-md px-2 py-1 text-xs" key={run.id}>
                <span className="truncate text-foreground/85">
                  {run.title?.trim() || run.preview?.trim() || run.id}
                </span>
                <span className="shrink-0 text-[0.62rem] text-muted-foreground/55 tabular-nums">
                  {formatRunTime(run.last_active || run.started_at)}
                </span>
              </div>
            ) : (
              // One click to the run's transcript; a run the scheduler never
              // closed opens view-only (see `openCronRun`, #88443).
              <button
                className="row-hover flex items-center justify-between gap-3 rounded-md px-2 py-1 text-left text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                key={run.id}
                onClick={onOpenSession ? () => openCronRun(run, onOpenSession) : undefined}
                type="button"
              >
                <span className="truncate text-foreground/85">
                  {run.title?.trim() || run.preview?.trim() || run.id}
                </span>
                <span className="shrink-0 text-[0.62rem] text-muted-foreground/55 tabular-nums">
                  {formatRunTime(run.last_active || run.started_at)}
                </span>
              </button>
            )
          )}
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

function EditorError({ error }: { error: null | string }) {
  return error ? (
    <div className="flex items-start gap-2 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
      <span>{error}</span>
    </div>
  ) : null
}

// The form's starting values: a parked draft wins, else the job being edited,
// else a blank daily job.
function editorSeed(editor: EditorState): CronEditorDraftValues {
  if (editor.mode !== 'closed' && editor.draft) {
    return editor.draft
  }

  const job = editor.mode === 'edit' ? editor.job : null

  if (!job) {
    return {
      carried: {},
      deliver: DEFAULT_DELIVER,
      modelChoice: MODEL_DEFAULT_VALUE,
      name: '',
      prompt: '',
      schedule: SCHEDULE_OPTIONS[0].expr ?? '',
      schedulePreset: SCHEDULE_OPTIONS[0].value
    }
  }

  return {
    carried: {},
    deliver: jobDeliver(job),
    modelChoice: jobModel(job) ? cronModelChoiceValue(jobProvider(job), jobModel(job)) : MODEL_DEFAULT_VALUE,
    name: jobName(job),
    prompt: jobPrompt(job),
    schedule: jobScheduleExpr(job),
    schedulePreset: scheduleOptionForExpr(jobScheduleExpr(job)).value
  }
}

// A new job that runs like `job`: its prompt, schedule, delivery and model in the
// editor, plus the settings the editor doesn't show.
function copyOfJob(job: CronJob, c: Translations['cron']): CronEditorDraftValues {
  return {
    ...editorSeed({ job, mode: 'edit' }),
    carried: carriedFields(job),
    name: c.blueprints.copyName(jobTitle(job))
  }
}

// A rendered blueprint as editable form values. Desktop has no origin chat, so
// an "origin" delivery becomes This desktop, as the blueprint form does.
function draftFromSpec(spec: AutomationBlueprintJobSpec): CronEditorDraftValues {
  const deliver = spec.deliver && spec.deliver !== 'origin' ? spec.deliver : DEFAULT_DELIVER

  return {
    carried: carriedFields(spec),
    deliver,
    modelChoice: MODEL_DEFAULT_VALUE,
    name: spec.name ?? '',
    prompt: spec.prompt,
    schedule: spec.schedule,
    schedulePreset: scheduleOptionForExpr(spec.schedule).value
  }
}

function CronEditorDialog({
  blueprintProfile,
  editor,
  jobs,
  onBlueprintCreate,
  onClose,
  onOpenMessaging,
  onSave,
  onTestPrompt
}: {
  blueprintProfile: string
  editor: EditorState
  /** Jobs "Start from" can copy. */
  jobs: readonly CronJob[]
  onBlueprintCreate: (blueprint: AutomationBlueprint, values: Record<string, string>) => Promise<void>
  onClose: () => void
  onOpenMessaging?: (draft: CronEditorDraftValues) => void
  onSave: (values: EditorValues) => Promise<void>
  onTestPrompt?: (draft: CronEditorDraftValues) => void
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
  // Per-job model override encoded as an opaque provider/model pair.
  // MODEL_DEFAULT_VALUE = follow the global default.
  const [modelChoice, setModelChoice] = useState(MODEL_DEFAULT_VALUE)
  // Blueprint fills typed slots (time/enum/weekdays/text) instead of the raw
  // cron fields; the backend renders the prompt + schedule from them.
  const [slotValues, setSlotValues] = useState<Record<string, string>>({})
  // Create mode can start blank, from a blueprint (its typed slots replace the
  // form until "Customize prompt"), or from a copy of a job (see start-from.tsx).
  const [templateChoice, setTemplateChoice] = useState(BLANK_START)
  const [carried, setCarried] = useState<CronEditorDraftValues['carried']>({})
  // The blueprint title a customized form came from, for the "Start from" hint.
  const [customizedFrom, setCustomizedFrom] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // The blueprint catalog powers the create dialog's "Start from" dropdown; it's
  // meaningless when editing an existing job, so skip the fetch there.
  const blueprintsQuery = useQuery({
    queryKey: ['cron-blueprints', blueprintProfile],
    queryFn: async () => (await getAutomationBlueprints(blueprintProfile)).blueprints,
    enabled: open && !isEdit
  })

  const blueprintList = blueprintsQuery.data ?? []

  const blueprint = blueprintList.find(item => item.key === templateChoice) ?? null

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

    fillForm(editorSeed(editor))
    setSlotValues({})
    setTemplateChoice(editor.mode === 'create' ? (editor.blueprintKey ?? BLANK_START) : BLANK_START)
    setCustomizedFrom(null)
    setError(null)
    setSaving(false)
  }, [editor, open])

  // Seed the typed slots with the blueprint's defaults whenever a blueprint is
  // picked from "Start from" (and reset them when switching back to Custom).
  useEffect(() => {
    setSlotValues(blueprint ? initialBlueprintValues(blueprint) : {})
    setError(null)
  }, [blueprint])

  function fillForm(values: CronEditorDraftValues) {
    setCarried(values.carried)
    setDeliver(values.deliver)
    setModelChoice(values.modelChoice)
    setName(values.name)
    setPrompt(values.prompt)
    setSchedule(values.schedule)
    setSchedulePreset(values.schedulePreset)
  }

  const draftValues = (): CronEditorDraftValues => ({
    carried,
    deliver,
    modelChoice,
    name,
    prompt,
    schedule,
    schedulePreset
  })

  function chooseStartFrom(value: string) {
    const job = startJob(value, jobs)

    fillForm(job ? copyOfJob(job, c) : editorSeed({ mode: 'create' }))
    setCustomizedFrom(null)
    setTemplateChoice(value)
  }

  // Open the blueprint, as its slots are filled now, in the full editor.
  async function customizeBlueprint() {
    if (!blueprint) {
      return
    }

    setError(null)

    try {
      const spec = await renderAutomationBlueprint({ blueprint: blueprint.key, values: slotValues }, blueprintProfile)

      fillForm(draftFromSpec(spec))
      setCustomizedFrom(blueprint.title)
      setTemplateChoice(BLANK_START)
    } catch (err) {
      setError(cleanBlueprintFieldError(err instanceof Error ? err.message : String(err)))
    }
  }

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
    modelProviders.some(provider =>
      (provider.models ?? []).some(model => cronModelChoiceValue(provider.slug, model) === modelChoice)
    )

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

    const override = parseCronModelChoiceValue(modelChoice)

    setSaving(true)
    setError(null)

    try {
      await onSave({
        carried,
        deliver,
        model: override?.model ?? '',
        name: name.trim(),
        prompt: prompt.trim(),
        provider: override?.provider ?? '',
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
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{isEdit ? c.editTitle : c.createTitle}</DialogTitle>
          <DialogDescription>{isEdit ? c.editDesc : c.createDesc}</DialogDescription>
        </DialogHeader>

        {!isEdit && (
          <StartFromField
            blueprints={blueprintList}
            c={c}
            customizedFrom={customizedFrom}
            jobs={copyableJobs(jobs)}
            onChange={chooseStartFrom}
            value={templateChoice}
          />
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

            <EditorError error={error} />

            <DialogFooter>
              <Tip label={c.blueprints.customizeHint}>
                <Button
                  className="sm:mr-auto"
                  disabled={saving}
                  onClick={() => void customizeBlueprint()}
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="edit" size="0.8rem" />
                  {c.blueprints.customize}
                </Button>
              </Tip>
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
                className="max-h-[50vh] min-h-56 resize-y font-mono"
                id="cron-prompt"
                onChange={event => setPrompt(event.target.value)}
                placeholder={c.promptPlaceholder}
                value={prompt}
              />
              {carried.skills && <FieldHint>{c.runsWithSkills(carried.skills.join(', '))}</FieldHint>}
            </Field>

            <ScheduleFields
              c={c}
              onChange={next => {
                setSchedulePreset(next.preset)
                setSchedule(next.schedule)
                setError(null)
              }}
              value={{ preset: schedulePreset, schedule }}
            />

            <Field htmlFor="cron-deliver" label={c.deliverLabel}>
              <DeliverCheckboxes
                c={c}
                id="cron-deliver"
                onChange={setDeliver}
                targets={deliveryTargets.data ?? []}
                value={deliver}
              />
              {/* Delivery lists connected platforms only; say so, so a missing
                  Email or SMS reads as "not connected", not "not supported". */}
              <FieldHint>
                {c.deliverConnectHint}{' '}
                {onOpenMessaging && (
                  <Button onClick={() => onOpenMessaging(draftValues())} size="inline" type="button" variant="link">
                    {c.deliverConnectAction}
                  </Button>
                )}
              </FieldHint>
            </Field>

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
                        {parseCronModelChoiceValue(modelChoice)?.model ?? modelChoice}
                      </SelectItem>
                    )}
                    {modelProviders.map(provider => (
                      <SelectGroup key={provider.slug}>
                        <SelectLabel>{provider.name}</SelectLabel>
                        {(provider.models ?? []).map(model => {
                          const value = cronModelChoiceValue(provider.slug, model)

                          return (
                            <SelectItem className="font-mono" key={value} value={value}>
                              {model}
                            </SelectItem>
                          )
                        })}
                      </SelectGroup>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}

            <EditorError error={error} />

            <DialogFooter>
              {onTestPrompt && !scriptOnlyJob && (
                // Runs nothing on its own: a fresh chat with the prompt typed in,
                // so the user sees the result before the schedule goes live.
                <Tip label={c.testInChatHint}>
                  <Button
                    className="sm:mr-auto"
                    disabled={saving || !prompt.trim()}
                    onClick={() => onTestPrompt(draftValues())}
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name="play" size="0.8rem" />
                    {c.testInChat}
                  </Button>
                </Tip>
              )}
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

// `draft` seeds the form from a parked draft instead of the job / blank defaults.
type EditorState =
  | { draft?: CronEditorDraftValues; job: CronJob; mode: 'edit' }
  | { mode: 'closed' }
  // `blueprintKey` pre-selects a blueprint in the create dialog's "Start from"
  // dropdown (set when a recipe row in the list rail is clicked).
  | { blueprintKey?: string; draft?: CronEditorDraftValues; mode: 'create' }

interface EditorValues {
  carried: CronEditorDraftValues['carried']
  deliver: string
  /** Per-job model override ('' = follow the global default). */
  model: string
  name: string
  prompt: string
  /** Provider slug for the model override ('' = none). */
  provider: string
  schedule: string
}
