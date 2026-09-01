import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { NEW_CHAT_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import {
  activateLocalModel,
  deleteLocalModel,
  downloadBrowsedModel,
  downloadLocalModel,
  ejectLocalModel,
  getLocalCatalog,
  getLocalHardware,
  getLocalModelsStatus,
  type HFFileGroup,
  type HFSearchHit,
  installLocalRuntime,
  listHFRepoFiles,
  quickstartLocalModels,
  searchHFModels,
  setLocalServer,
  sideloadLocalModel
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, CheckCircle2, Cpu, Download, Eject, FolderOpen, Loader2, Monitor, Package, Search, StopFilled, Trash2, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $localRuntimeJobs,
  runningDownloadFor,
  runningRuntimeInstall,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'
import type { LocalCatalogModel, LocalHardware, LocalModelsStatus } from '@/types/hermes'

import { ListRow, Pill, SettingsContent, SettingsSection, SettingsSkeleton } from './primitives'

function ProgressBar({ percent }: { percent: number | undefined }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
      <div
        className="h-full rounded-full bg-primary transition-[width] duration-300"
        style={{ width: `${Math.max(2, Math.min(100, percent ?? 2))}%` }}
      />
    </div>
  )
}

function gbLabel(bytes: number | null | undefined): string {
  if (!bytes) {
    return '—'
  }

  return `${(bytes / (1 << 30)).toFixed(1)} GB`
}

// Catalog display order: what runs well leads. Resident (all on GPU)
// first, then spilled (works, slower), then doesn't-fit; catalog order
// (recommended first) holds within each band.
function fitRank(model: LocalCatalogModel): number {
  if (model.fits && !model.spilled) {
    return 0
  }

  if (model.fits) {
    return 1
  }

  return 2
}

export function LocalModelsSettings() {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [status, setStatus] = useState<LocalModelsStatus | null>(null)
  const [hardware, setHardware] = useState<LocalHardware | null>(null)
  const [catalog, setCatalog] = useState<LocalCatalogModel[] | null>(null)
  const [deleting, setDeleting] = useState<null | string>(null)
  const [serverBusy, setServerBusy] = useState(false)
  // Quickstart escape hatch: true once the user asks for the full pane
  // (model list, HF browser) instead of the one-button setup card.
  const [configure, setConfigure] = useState(false)
  // Jobs live in the app-level store (they must survive this pane
  // unmounting); the pane just renders the slice it cares about.
  const jobs = useStore($localRuntimeJobs)

  const refresh = useCallback(() => {
    void getLocalModelsStatus()
      .then(setStatus)
      .catch(() => setStatus(null))
    void getLocalCatalog()
      .then(data => setCatalog(data.models))
      .catch(() => setCatalog([]))
  }, [])

  // Snappy first paint: status + catalog immediately; hardware (may shell out
  // to nvidia-smi) backfills and pops in-place. The job watcher also kicks
  // here so reopening the pane rediscovers work started before.
  useEffect(() => {
    refresh()
    watchLocalRuntimeJobs()
    void getLocalHardware()
      .then(setHardware)
      .catch(() => setHardware(null))
  }, [refresh])

  // The pane is LIVE while visible: residency changes without user action
  // (boot warm finishing, idle sweep unloading, another surface ejecting),
  // and a stale snapshot here reads as a broken feature — 'VRAM full but
  // the pane says Not in memory'. The status route is built cheap for
  // polling; setTimeout chain, never overlapping.
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const tick = async () => {
      try {
        const next = await getLocalModelsStatus()

        if (!cancelled) {
          setStatus(next)
        }
      } catch {
        // Backend briefly unreachable — keep the last snapshot.
      }

      if (!cancelled) {
        timer = window.setTimeout(() => void tick(), 4_000)
      }
    }

    timer = window.setTimeout(() => void tick(), 4_000)

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [])

  // A job finishing (download done, install done) changes what status/catalog
  // should show — refresh whenever the running set shrinks.
  const runningCount = jobs.filter(j => j.status === 'running').length
  useEffect(() => {
    refresh()
  }, [refresh, runningCount])

  async function handleInstallRuntime() {
    try {
      await installLocalRuntime()
      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.installFailed)
    }
  }

  async function handleQuickstart() {
    try {
      await quickstartLocalModels()
      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.quickstartFailed)
    }
  }

  async function handleDownload(model: LocalCatalogModel) {
    try {
      const res = await downloadLocalModel(model.id)

      if (res.already_downloaded || !res.job_id) {
        refresh()

        return
      }

      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.downloadFailed(model.display_name))
    }
  }

  async function handleActivate(target: null | string, displayName: string) {
    if (!target) {
      return
    }

    try {
      await activateLocalModel(target)
      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.activateFailed(displayName))
    }
  }

  async function handleEject(modelId: string) {
    try {
      await ejectLocalModel(modelId)
      notify({ durationMs: 3_000, kind: 'success', message: copy.ejected, title: copy.title })
      refresh()
    } catch (err) {
      notifyError(err, copy.ejectFailed)
    }
  }

  async function handleServer(action: 'start' | 'stop') {
    setServerBusy(true)

    try {
      await setLocalServer(action)
      notify({
        durationMs: 3_500,
        kind: 'success',
        message: action === 'stop' ? copy.serverStopped : copy.serverStarted,
        title: copy.title
      })
      refresh()
    } catch (err) {
      notifyError(err, action === 'stop' ? copy.serverStopFailed : copy.serverStartFailed)
    } finally {
      setServerBusy(false)
    }
  }

  async function handleDelete(target: string, rowId: string) {
    if (!window.confirm(copy.deleteConfirm(target))) {
      return
    }

    setDeleting(rowId)

    try {
      await deleteLocalModel(target)
      notify({ durationMs: 2_500, kind: 'success', message: copy.deleted(target), title: copy.title })
      refresh()
    } catch (err) {
      notifyError(err, copy.deleteFailed)
    } finally {
      setDeleting(null)
    }
  }

  // Setup flows end at the action, not the settings pane: when quickstart
  // finishes while the user is still HERE watching it, land them on a new
  // chat with the model ready to try. Unmount cancels the intent — a user
  // who navigated away mid-download keeps their place (no focus theft).
  // (Lives above the loading return: hooks run unconditionally.)
  const navigate = useNavigate()
  const seenQuickstarts = useRef(new Set<string>())

  const runningQuickstart = jobs.find(
    j => j.kind === 'quickstart' && j.status === 'running'
  )

  useEffect(() => {
    // Event detection, not value mirroring: the ref only remembers which
    // job ids THIS mount saw running, so a 'done' already in the list on
    // mount (stale history) never triggers a navigation.
    const seen = seenQuickstarts.current

    for (const j of jobs) {
      if (j.kind !== 'quickstart') {
        continue
      }

      if (j.status === 'running') {
        seen.add(j.job_id)
      } else if (j.status === 'done' && seen.has(j.job_id)) {
        seen.delete(j.job_id)
        navigate(NEW_CHAT_ROUTE)
      }
    }
  }, [jobs, navigate])

  if (!status || catalog === null) {
    return <SettingsSkeleton sections={[{ rows: 2 }, { rows: 4 }]} />
  }

  const rJob = runningRuntimeInstall(jobs)
  const lastError = jobs.find(j => j.status === 'error')

  const sortedCatalog = [...catalog].sort((a, b) => fitRank(a) - fitRank(b))

  // ── Quickstart: the dummy-proof front door ──
  // Until something is servable (runtime + at least one model), the pane
  // leads with a hero that does everything in one click; the full pane
  // stays one 'Configure…' click away. A running quickstart pins this
  // view so its progress has a home even after a remount.
  const qJob = runningQuickstart ?? null

  const needsSetup = !status.runtime_installed || status.models.length === 0
  const heroModel = catalog.find(c => c.recommended && c.fits) ?? catalog.find(c => c.fits) ?? null

  if (qJob || (needsSetup && !configure && heroModel)) {
    // Stage rail derived from the job phase: engine -> model -> finish.
    const phase = qJob?.phase ?? ''

    const stageIndex = ['starting-server', 'setting-default'].includes(phase)
      ? 2
      : phase === 'downloading'
        ? 1
        : 0

    const stages = [copy.quickstartStageEngine, copy.quickstartStageModel, copy.quickstartStageFinish]

    // The model-download leg blanks job.detail on purpose (pane rows
    // render their own byte counter) — compose one here instead of
    // falling back to runtime copy that would misname the stage.
    const liveDetail =
      qJob &&
      (qJob.detail ||
        (qJob.total_bytes
          ? copy.downloadProgress(gbLabel(qJob.done_bytes), gbLabel(qJob.total_bytes))
          : copy.installing))

    return (
      <SettingsContent>
        <div className="flex min-h-[60dvh] items-center justify-center">
          <div className="w-full max-w-md text-center">
            <div className="mx-auto mb-5 flex size-14 items-center justify-center rounded-2xl bg-primary/10">
              {qJob ? (
                <Loader2 className="size-7 animate-spin text-primary" />
              ) : (
                <Cpu className="size-7 text-primary" />
              )}
            </div>

            <h2 className="text-lg font-semibold text-foreground">
              {qJob ? qJob.target : (heroModel?.display_name ?? '')}
            </h2>

            {qJob ? (
              <>
                <p className="mt-2 min-h-10 text-[0.8rem] leading-5 text-muted-foreground">
                  {liveDetail}
                </p>

                <div className="mt-5">
                  <ProgressBar percent={qJob.percent} />
                </div>

                {/* Stage rail: engine -> model -> finish. */}
                <div className="mt-5 flex items-center justify-center gap-5">
                  {stages.map((label, i) => (
                    <span
                      className={cn(
                        'inline-flex items-center gap-1.5 text-[0.72rem]',
                        i < stageIndex && 'text-(--ui-text-tertiary)',
                        i === stageIndex && 'font-medium text-foreground',
                        i > stageIndex && 'text-(--ui-text-tertiary) opacity-60'
                      )}
                      key={label}
                    >
                      {i < stageIndex ? (
                        <CheckCircle2 className="size-3.5 text-primary" />
                      ) : i === stageIndex ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <span className="size-1.5 rounded-full bg-current" />
                      )}
                      {label}
                    </span>
                  ))}
                </div>
              </>
            ) : heroModel ? (
              <>
                <p className="mt-2 text-[0.8rem] leading-5 text-muted-foreground">
                  {heroModel.downloaded
                    ? copy.quickstartDetailReady(heroModel.display_name)
                    : copy.quickstartDetail(heroModel.display_name, heroModel.size_label)}
                </p>

                <div className="mt-6 flex items-center justify-center gap-3">
                  <Button onClick={() => setConfigure(true)} size="sm" variant="outline">
                    {copy.quickstartConfigure}
                  </Button>
                  <Button onClick={() => void handleQuickstart()} size="default">
                    <Zap />
                    {copy.quickstartAction}
                  </Button>
                </div>
              </>
            ) : null}

            {lastError?.kind === 'quickstart' && !qJob && (
              <p className="mt-4 text-[0.75rem] text-destructive">{lastError.error}</p>
            )}
          </div>
        </div>
      </SettingsContent>
    )
  }

  // Up to date = the authority (status) says the configured tag is what's
  // serving. Shown whenever true — not only right after an update.
  const updateApplied =
    status.runtime_installed && !status.update_available && status.tag === status.configured_tag

  return (
    <SettingsContent>
      {/* ── Runtime ── */}
      <SettingsSection
        aside={
          status.runtime_installed ? (
            <Pill tone="primary">
              {status.server_running ? copy.serverRunning : copy.runtimeReady(status.runtime_backend ?? '')}
            </Pill>
          ) : undefined
        }
        icon={Zap}
        meta={status.tag}
        title={copy.runtimeTitle}
      >
        {status.runtime_installed ? (
          <ListRow
            action={
              status.server_running ? (
                <Button
                  className={cn(serverBusy && '[&_svg]:animate-spin')}
                  disabled={serverBusy}
                  onClick={() => void handleServer('stop')}
                  size="sm"
                  variant="outline"
                >
                  {serverBusy ? <Loader2 /> : <StopFilled />}
                  {copy.stopServer}
                </Button>
              ) : (
                <Button
                  className={cn(serverBusy && '[&_svg]:animate-spin')}
                  disabled={serverBusy}
                  onClick={() => void handleServer('start')}
                  size="sm"
                  variant="outline"
                >
                  {serverBusy ? <Loader2 /> : <Zap />}
                  {copy.startServer}
                </Button>
              )
            }
            description={
              status.server_running
                ? copy.runtimeRunningDetail
                : copy.runtimeInstalledDetail(status.tag, status.runtime_backend ?? 'cpu')
            }
            title={copy.runtimeInstalled}
          />
        ) : rJob ? (
          <ListRow
            below={<ProgressBar percent={rJob.percent} />}
            description={rJob.detail || copy.installing}
            title={
              <span className="inline-flex items-center gap-2">
                <Loader2 className="size-3.5 animate-spin" />
                {copy.installing}
              </span>
            }
          />
        ) : (
          <ListRow
            action={
              <Button onClick={() => void handleInstallRuntime()} size="sm">
                <Download />
                {copy.installAction}
              </Button>
            }
            description={copy.installDetail}
            title={copy.installTitle}
          />
        )}

        {status.update_available && !rJob && (
          <ListRow
            action={
              <Button onClick={() => void handleInstallRuntime()} size="sm">
                <Download />
                {copy.updateAction}
              </Button>
            }
            description={copy.updateDetail(status.configured_tag, status.tag)}
            title={copy.updateTitle}
          />
        )}

        {rJob && status.runtime_installed && (
          <ListRow
            below={<ProgressBar percent={rJob.percent} />}
            description={rJob.detail || copy.updating}
            title={
              <span className="inline-flex items-center gap-2">
                <Loader2 className="size-3.5 animate-spin" />
                {copy.updating}
              </span>
            }
          />
        )}

        {updateApplied && (
          <ListRow
            description={copy.upToDateDetail(status.tag, status.runtime_backend ?? 'cpu')}
            title={
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 dark:text-emerald-400" />
                {copy.upToDateTitle}
              </span>
            }
          />
        )}

        {lastError?.kind === 'runtime-install' && (
          <p className="text-[0.75rem] text-destructive">{lastError.error}</p>
        )}
      </SettingsSection>

      {/* ── This machine ── */}
      <SettingsSection icon={Monitor} title={copy.hardwareTitle}>
        {hardware ? (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            {hardware.gpu_name && (
              <span className="inline-flex items-center gap-1.5">
                <Zap className="size-3.5" />
                {hardware.gpu_name}
              </span>
            )}

            <span className="inline-flex items-center gap-1.5">
              <Cpu className="size-3.5" />
              {copy.vram(gbLabel(hardware.vram_total_bytes))}
            </span>

            <span className="inline-flex items-center gap-1.5">
              <Package className="size-3.5" />
              {copy.ram(gbLabel(hardware.ram_total_bytes))}
            </span>

            {hardware.uma && <Pill>{copy.unifiedMemory}</Pill>}
          </div>
        ) : (
          <p className="py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            {copy.hardwareLoading}
          </p>
        )}
      </SettingsSection>

      {/* ── Models ── */}
      <SettingsSection icon={Download} meta={`${catalog.length}`} title={copy.modelsTitle}>
        <div className="grid gap-1">
          {sortedCatalog.map(model => {
            const dJob = runningDownloadFor(jobs, model.id)
            const anyDownloadRunning = jobs.some(j => j.kind === 'model-download' && j.status === 'running')
            const activateTarget = model.downloaded_model_id ?? model.model_id
            const isActive = Boolean(activateTarget && status.active_model_id === activateTarget)
            const residency = activateTarget ? status.loaded_models[activateTarget] : undefined
            const isLoaded = residency === 'loaded' || residency === 'ready'
            const isLoadingNow = residency === 'loading'
            const livePlacement = activateTarget ? status.placement?.[activateTarget] : undefined

            const aJob = jobs.find(
              j => j.kind === 'model-activate' && j.status === 'running' && j.model_id === activateTarget
            )

            const anyActivateRunning = jobs.some(j => j.kind === 'model-activate' && j.status === 'running')

            return (
              <ListRow
                action={
                  model.downloaded ? (
                    <div className="flex items-center justify-end gap-2">
                      {isLoaded && livePlacement && (
                        <Tip
                          label={
                            livePlacement.spilled
                              ? copy.placementSpilledTip
                              : copy.placementResidentTip
                          }
                        >
                          <Pill tone={livePlacement.spilled ? 'warn' : 'success'}>
                            <Cpu className="mr-1 size-3" />
                            {livePlacement.granted_window_label ?? livePlacement.window_label ?? ''}
                            {' · '}
                            {livePlacement.spilled ? copy.placementSpilled : copy.placementResident}
                          </Pill>
                        </Tip>
                      )}
                      {isLoaded && !livePlacement && <Pill>{copy.loadedPill}</Pill>}

                      {isLoadingNow && (
                        <Pill>
                          <Loader2 className="mr-1 size-3 animate-spin" />
                          {copy.loadingPill}
                        </Pill>
                      )}

                      {isActive ? (
                        <Tip label={copy.activeDetail}>
                          <Pill tone="primary">
                            <Check className="mr-1 size-3" />
                            {copy.activePill}
                          </Pill>
                        </Tip>
                      ) : (
                        <Button
                          className={cn(aJob && '[&_svg]:animate-spin')}
                          disabled={anyActivateRunning}
                          onClick={() => void handleActivate(activateTarget ?? null, model.display_name)}
                          size="sm"
                        >
                          {aJob ? <Loader2 /> : <Check />}
                          {aJob ? copy.activating : copy.useAction}
                        </Button>
                      )}

                      {isLoaded && (
                        <Tip label={copy.ejectTip}>
                          <Button
                            onClick={() => void handleEject(activateTarget ?? model.id)}
                            size="icon"
                            variant="ghost"
                          >
                            <Eject />
                          </Button>
                        </Tip>
                      )}

                      <Tip label={copy.deleteAction}>
                        <Button
                          className={cn(deleting === model.id && '[&_svg]:animate-spin')}
                          onClick={() => void handleDelete(model.downloaded_model_id ?? model.id, model.id)}
                          size="icon"
                          variant="ghost"
                        >
                          {deleting === model.id ? <Loader2 /> : <Trash2 />}
                        </Button>
                      </Tip>
                    </div>
                  ) : dJob ? undefined : (
                    <Button
                      disabled={!model.fits || anyDownloadRunning || !status.runtime_installed}
                      onClick={() => void handleDownload(model)}
                      size="sm"
                      variant="outline"
                    >
                      <Download />
                      {copy.downloadAction(model.size_label)}
                    </Button>
                  )
                }
                below={
                  dJob ? (
                    <div className="mt-2 grid gap-1">
                      <ProgressBar percent={dJob.percent} />

                      <p className="text-[0.68rem] text-muted-foreground">
                        {!dJob.done_bytes && dJob.detail
                          ? dJob.detail
                          : copy.downloadProgress(gbLabel(dJob.done_bytes), gbLabel(dJob.total_bytes))}
                      </p>
                    </div>
                  ) : undefined
                }
                description={
                  <>
                    {model.description}

                    <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      {/* Memory: the traffic light. Green = runs fully on
                          the GPU; amber = spills to system RAM (works,
                          slower); red = doesn't fit this machine at all.
                          Detail prose lives in the tooltip. */}
                      {!model.fits ? (
                        <Tip label={model.fit_detail ?? model.fit_summary}>
                          <Pill tone="destructive">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillTooBig}
                          </Pill>
                        </Tip>
                      ) : model.spilled ? (
                        <Tip label={model.quant_reason ?? model.fit_summary}>
                          <Pill tone="warn">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillUsesRam}
                          </Pill>
                        </Tip>
                      ) : (
                        <Tip label={model.quant_reason ?? model.fit_summary}>
                          <Pill tone="success">
                            <Cpu className="mr-1 size-3" />
                            {copy.pillFitsGpu}
                          </Pill>
                        </Tip>
                      )}

                      {/* Context: one pill. Green 'Full X context' only when
                          the model earned its complete window resident on the
                          GPU — a big context served from system RAM is slow,
                          and a green badge there would sell exactly the wrong
                          model, so a spilled full window goes gray. Anything
                          starting below its native window gets one quiet
                          'Up to' pill instead of a start/grow pair. */}
                      {model.fits && model.start_window_label && (
                        model.start_window && model.start_window >= model.native_context ? (
                          <Tip label={copy.pillFullContextTip}>
                            <Pill tone={model.spilled ? 'muted' : 'success'}>
                              {copy.pillFullContext(model.native_context_label)}
                            </Pill>
                          </Tip>
                        ) : (
                          <Tip label={copy.pillGrowsTip}>
                            <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>
                          </Tip>
                        )
                      )}

                      {!model.fits && (
                        <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>
                      )}

                      {model.vision && <Pill>{copy.pillVision}</Pill>}
                    </span>

                    {isActive && !isLoaded && !isLoadingNow && status.server_running && (
                      <span className="mt-0.5 block text-(--ui-text-tertiary)">{copy.activeNotLoaded}</span>
                    )}
                  </>
                }
                key={model.id}
                title={
                  <span className="inline-flex items-center gap-2">
                    {model.display_name}

                    {model.recommended &&
                      (model.recommended_reason ? (
                        // The why, straight from the resolver: the tooltip is
                        // the branch that picked this model, so the shown
                        // rationale can never drift from the actual decision.
                        <Tip label={copy.recommendedReason[model.recommended_reason]}>
                          <Pill tone="primary">{copy.recommended}</Pill>
                        </Tip>
                      ) : (
                        <Pill tone="primary">{copy.recommended}</Pill>
                      ))}
                  </span>
                }
              />
            )
          })}

          {status.models
            .filter(m => !catalog.some(c => c.downloaded_model_id === m.id || c.model_id === m.id))
            .map(m => {
              const isActive = status.active_model_id === m.id
              const residency = status.loaded_models[m.id]
              const isLoaded = residency === 'loaded' || residency === 'ready'
              const isLoadingNow = residency === 'loading'
              const livePlacement = status.placement?.[m.id]

              const aJob = jobs.find(
                j => j.kind === 'model-activate' && j.status === 'running' && j.model_id === m.id
              )

              const anyActivateRunning = jobs.some(j => j.kind === 'model-activate' && j.status === 'running')

              return (
                <ListRow
                  action={
                    <div className="flex items-center justify-end gap-2">
                      {isLoaded && livePlacement && (
                        <Tip
                          label={livePlacement.spilled ? copy.placementSpilledTip : copy.placementResidentTip}
                        >
                          <Pill tone={livePlacement.spilled ? 'warn' : 'success'}>
                            <Cpu className="mr-1 size-3" />
                            {livePlacement.granted_window_label ?? livePlacement.window_label ?? ''}
                            {' · '}
                            {livePlacement.spilled ? copy.placementSpilled : copy.placementResident}
                          </Pill>
                        </Tip>
                      )}
                      {isLoaded && !livePlacement && <Pill>{copy.loadedPill}</Pill>}

                      {isLoadingNow && (
                        <Pill>
                          <Loader2 className="mr-1 size-3 animate-spin" />
                          {copy.loadingPill}
                        </Pill>
                      )}

                      {isActive ? (
                        <Pill tone="primary">
                          <CheckCircle2 className="mr-1 size-3" />
                          {copy.activePill}
                        </Pill>
                      ) : (
                        <Button
                          className={cn(aJob && '[&_svg]:animate-spin')}
                          disabled={anyActivateRunning}
                          onClick={() => void handleActivate(m.id, m.id)}
                          size="sm"
                        >
                          {aJob ? <Loader2 /> : <Check />}
                          {copy.useAction}
                        </Button>
                      )}

                      {isLoaded && (
                        <Tip label={copy.ejectTip}>
                          <Button onClick={() => void handleEject(m.id)} size="icon" variant="ghost">
                            <Eject />
                          </Button>
                        </Tip>
                      )}

                      <Tip label={copy.deleteAction}>
                        <Button
                          className={cn(deleting === m.id && '[&_svg]:animate-spin')}
                          onClick={() => void handleDelete(m.id, m.id)}
                          size="icon"
                          variant="ghost"
                        >
                          {deleting === m.id ? <Loader2 /> : <Trash2 />}
                        </Button>
                      </Tip>
                    </div>
                  }
                  description={<span>{copy.addedByYou}</span>}
                  key={m.id}
                  title={
                    <span className="inline-flex items-center gap-2">
                      <span className="truncate font-mono text-[0.8rem]">{m.id}</span>

                      <span className="text-[0.68rem] font-normal text-muted-foreground">{m.size_label}</span>
                    </span>
                  }
                />
              )
            })}
        </div>

        {lastError?.kind === 'model-download' && (
          <p className="text-[0.75rem] text-destructive">{lastError.error}</p>
        )}
      </SettingsSection>

      <BrowseSection onChanged={refresh} />
    </SettingsContent>
  )
}

function fitTone(fit: HFFileGroup['fit']): 'destructive' | 'muted' | 'success' | 'warn' {
  if (fit === 'fits-gpu') {
    return 'success'
  }

  if (fit === 'needs-ram') {
    return 'warn'
  }

  if (fit === 'too-big') {
    return 'destructive'
  }

  return 'muted'
}

function browsedModelId(group: HFFileGroup): string {
  // Mirrors the backend's derivation: first file's name, split-part
  // suffix stripped — the id the download job carries.
  const first = group.paths[0].split('/').pop() ?? group.paths[0]

  return first.replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '')
}

function BrowseSection({ onChanged }: { onChanged: () => void }) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const jobs = useStore($localRuntimeJobs)
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<HFSearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [openRepo, setOpenRepo] = useState<null | string>(null)
  const [files, setFiles] = useState<HFFileGroup[]>([])
  const [listing, setListing] = useState(false)
  const [error, setError] = useState<null | string>(null)
  // Guard against the past: a stale search result must never overwrite a
  // newer query's hits (the desktop guide's out-of-order rule).
  const searchSeq = useRef(0)

  useEffect(() => {
    const q = query.trim()

    if (q.length < 2) {
      setHits([])
      setSearching(false)

      return
    }

    const seq = ++searchSeq.current
    setSearching(true)

    const handle = setTimeout(() => {
      searchHFModels(q)
        .then(r => {
          if (searchSeq.current === seq) {
            setHits(r.hits)
            setError(null)
          }
        })
        .catch((e: Error) => {
          if (searchSeq.current === seq) {
            setError(e.message)
          }
        })
        .finally(() => {
          if (searchSeq.current === seq) {
            setSearching(false)
          }
        })
    }, 350)

    return () => clearTimeout(handle)
  }, [query])

  const openFiles = useCallback((repo: string) => {
    setOpenRepo(repo)
    setFiles([])
    setListing(true)
    listHFRepoFiles(repo)
      .then(r => setFiles(r.files))
      .catch((e: Error) => setError(e.message))
      .finally(() => setListing(false))
  }, [])

  const startBrowsedDownload = useCallback(
    (repo: string, group: HFFileGroup) => {
      downloadBrowsedModel(repo, group.paths)
        .then(r => {
          if (r.already_downloaded) {
            notify({ durationMs: 3_000, kind: 'info', message: copy.browseAlreadyDownloaded, title: copy.browseTitle })

            return
          }

          // Same feedback loop as catalog downloads: the job store polls
          // and the tile renders live progress from it.
          watchLocalRuntimeJobs()
          notify({
            durationMs: 3_000,
            kind: 'info',
            message: copy.browseDownloadStarted.replace('{name}', r.model_id),
            title: copy.browseTitle
          })
          onChanged()
        })
        .catch((e: Error) => notifyError(e, copy.browseTitle))
    },
    [copy.browseAlreadyDownloaded, copy.browseDownloadStarted, copy.browseTitle, onChanged]
  )

  const sideload = useCallback(() => {
    window.hermesDesktop
      .selectPaths({ filters: [{ extensions: ['gguf'], name: 'GGUF models' }], title: copy.sideloadTitle })
      .then(paths => {
        if (!paths.length) {
          return
        }

        return sideloadLocalModel(paths[0]).then(r => {
          notify({
            durationMs: 3_000,
            kind: 'success',
            message: r.already_present ? copy.sideloadAlreadyPresent : copy.sideloadDone.replace('{name}', r.model_id),
            title: copy.browseTitle
          })
          onChanged()
        })
      })
      .catch((e: Error) => notifyError(e, copy.browseTitle))
  }, [copy.browseTitle, copy.sideloadAlreadyPresent, copy.sideloadDone, copy.sideloadTitle, onChanged])

  return (
    <SettingsSection
      aside={
        <Button onClick={sideload} size="sm" variant="outline">
          <FolderOpen className="mr-1 size-3.5" />
          {copy.sideloadButton}
        </Button>
      }
      icon={Search}
      title={copy.browseTitle}
    >
      <p className="text-[0.75rem] text-muted-foreground">{copy.browseHint}</p>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <input
          className="w-full rounded-md border border-(--ui-border) bg-transparent py-1.5 pl-8 pr-3 text-[0.8rem] outline-none placeholder:text-muted-foreground focus:border-primary"
          onChange={e => setQuery(e.target.value)}
          placeholder={copy.browsePlaceholder}
          value={query}
        />
      </div>

      {searching && (
        <p className="flex items-center gap-2 text-[0.75rem] text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          {copy.browseSearching}
        </p>
      )}

      {error && <p className="text-[0.75rem] text-destructive">{error}</p>}

      <div className="grid gap-1">
        {hits.map(hit => (
          <div key={hit.repo}>
            <ListRow
              action={
                <Button onClick={() => openFiles(hit.repo)} size="sm" variant="ghost">
                  {openRepo === hit.repo ? copy.browseRefresh : copy.browseShowFiles}
                </Button>
              }
              description={
                <span>
                  {Intl.NumberFormat().format(hit.downloads)} {copy.browseDownloads}
                  {' · '}
                  {Intl.NumberFormat().format(hit.likes)} {copy.browseLikes}
                  {hit.gated ? ` · ${copy.browseGated}` : ''}
                </span>
              }
              title={<span className="font-mono text-[0.8rem]">{hit.repo}</span>}
            />

            {openRepo === hit.repo && (
              <div className="ml-4 grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-1.5 border-l border-(--ui-border) py-1 pl-3">
                {listing && (
                  <p className="col-span-full flex items-center gap-2 py-1 text-[0.75rem] text-muted-foreground">
                    <Loader2 className="size-3 animate-spin" />
                    {copy.browseListing}
                  </p>
                )}

                {!listing && files.length === 0 && (
                  <p className="col-span-full py-1 text-[0.75rem] text-muted-foreground">{copy.browseNoGguf}</p>
                )}

                {files.map(group => {
                  const dJob = runningDownloadFor(jobs, browsedModelId(group))

                  return (
                    <div
                      className={cn(
                        'flex flex-col gap-1 rounded-md border border-(--ui-border) px-2.5 py-1.5',
                        group.fit === 'too-big' && 'opacity-45'
                      )}
                      key={group.label}
                    >
                      <span className="flex w-full items-center justify-between gap-2">
                        <span className="truncate font-mono text-[0.75rem]">
                          {group.label}
                          {group.paths.length > 1 ? ` ×${group.paths.length}` : ''}
                        </span>

                        <Button
                          aria-label={copy.browseDownloadAria.replace('{name}', group.label)}
                          className="h-6 shrink-0 px-2"
                          disabled={group.fit === 'too-big' || Boolean(dJob)}
                          onClick={() => startBrowsedDownload(hit.repo, group)}
                          size="sm"
                          variant="ghost"
                        >
                          {dJob ? <Loader2 className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
                        </Button>
                      </span>

                      {dJob ? (
                        <>
                          <ProgressBar percent={dJob.percent} />

                          <span className="text-[0.68rem] text-muted-foreground">
                            {!dJob.done_bytes && dJob.detail
                              ? dJob.detail
                              : copy.downloadProgress(gbLabel(dJob.done_bytes), gbLabel(dJob.total_bytes))}
                          </span>
                        </>
                      ) : (
                        <span className="flex items-center justify-between gap-2">
                          <Pill tone={fitTone(group.fit)}>
                            <Cpu className="mr-1 size-3" />
                            {group.fit === 'fits-gpu'
                              ? copy.pillFitsGpu
                              : group.fit === 'needs-ram'
                                ? copy.pillUsesRam
                                : group.fit === 'too-big'
                                  ? copy.pillTooBig
                                  : copy.browseFitUnknown}
                          </Pill>

                          <span className="shrink-0 text-[0.7rem] text-muted-foreground">{gbLabel(group.total_bytes)}</span>
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </SettingsSection>
  )
}
