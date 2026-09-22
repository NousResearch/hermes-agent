import { atom } from 'nanostores'

import { getApiRequestConnection } from '@/api/client'
import { getLocalModelsJobs, installLocalRuntime } from '@/hermes'
import { translateNow } from '@/i18n'
import { $activeGatewayRoute } from '@/store/gateway'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'
import type { LocalRuntimeJob } from '@/types/hermes'

// App-level tracker for local-runtime jobs (runtime installs, model
// downloads). The AUTHORITY is the backend job registry — this store is a
// cache of it (desktop guide: server truth is cached, not owned). Living at
// the store layer, not in the settings pane, is what makes a download
// survive the pane unmounting: anything can start a job, the poller follows
// it to completion, and completion/failure notify app-wide exactly once.

export const $localRuntimeJobs = atom<readonly LocalRuntimeJob[]>([])

export const $localRuntimeInstallStarting = atom(false)

// Shared by the settings button and the campaign CTA. This is request state,
// not invented job progress; the backend registry still owns the actual work.
export function localRuntimeInstallBusy(): boolean {
  return (
    $localRuntimeInstallStarting.get() ||
    $localRuntimeJobs
      .get()
      .some(job => job.status === 'running' && (job.kind === 'runtime-install' || job.kind === 'quickstart'))
  )
}

export async function startLocalRuntimeInstall(): Promise<void> {
  if (localRuntimeInstallBusy()) {
    return
  }

  const owner = activeContext
  owner.postPending = true
  $localRuntimeInstallStarting.set(true)

  try {
    const { job_id } = await installLocalRuntime()
    owner.acceptedIds.add(job_id)
    owner.postPending = false
    owner.readPending = true
    owner.acceptedInstall++

    if (owner !== activeContext) {
      return
    }

    // The pane can mount while this POST resolves. Hold the shared lock
    // through a fresh read, not merely until the request was accepted.
    await polling

    if (owner !== activeContext) {
      return
    }

    await poll()
  } catch (error) {
    if (owner === activeContext) {
      notifyError(error, translateNow('settings.localModels.installFailed'))
    }
  } finally {
    owner.postPending = false

    if (owner === activeContext && !owner.readPending) {
      $localRuntimeInstallStarting.set(false)
    }
  }
}

const POLL_ACTIVE_MS = 700
let timer: null | ReturnType<typeof setTimeout> = null
let polling: Promise<void> | null = null
let generation = 0
interface JobContext {
  jobs: readonly LocalRuntimeJob[]
  postPending: boolean
  readPending: boolean
  acceptedInstall: number
  acceptedIds: Set<string>
  settledNotified: Set<string>
}

// Only foreground requests run; switching away retains ownership, not a poller.
const contexts = new Map<string, JobContext>()

function contextKey() {
  const connection = $connection.get()

  return JSON.stringify([
    getApiRequestConnection() ?? connection?.connectionId ?? [connection?.mode, connection?.baseUrl],
    $activeGatewayRoute.get()
  ])
}

function cachedContext(key: string): JobContext {
  let state = contexts.get(key)

  if (!state) {
    state = {
      jobs: [],
      postPending: false,
      readPending: false,
      acceptedInstall: 0,
      acceptedIds: new Set(),
      settledNotified: new Set()
    }
    contexts.set(key, state)
  }

  return state
}

let activeKey = contextKey()
let activeContext = cachedContext(activeKey)

function resetContext() {
  activeContext.jobs = $localRuntimeJobs.get()
  const context = ++generation

  if (timer !== null) {
    clearTimeout(timer)
  }

  timer = null
  polling = null
  activeKey = contextKey()
  activeContext = cachedContext(activeKey)
  activeContext.readPending ||= activeContext.jobs.some(job => job.status === 'running')
  $localRuntimeJobs.set(activeContext.jobs)
  $localRuntimeInstallStarting.set(activeContext.postPending || activeContext.readPending)

  // Gateway activation publishes the route BEFORE the REST profile tag.
  // Coalesce that synchronous re-home before any ambient API request.
  void Promise.resolve().then(() => {
    if (context !== generation) {
      return
    }

    if (activeKey !== contextKey()) {
      resetContext()

      return
    }

    if ($localModelsEnabled.get() && activeContext.readPending && !activeContext.postPending) {
      void poll()
    }
  })
}

$connection.listen(resetContext)
$activeGatewayRoute.listen(resetContext)

function jobsEqual(a: readonly LocalRuntimeJob[], b: readonly LocalRuntimeJob[]) {
  if (a.length !== b.length) {
    return false
  }

  return a.every((job, i) => {
    const other = b[i]

    return (
      job.job_id === other.job_id &&
      job.status === other.status &&
      job.phase === other.phase &&
      job.done_bytes === other.done_bytes
    )
  })
}

function notifySettled(previous: readonly LocalRuntimeJob[], next: readonly LocalRuntimeJob[]) {
  const { acceptedIds, settledNotified } = activeContext
  const wasRunning = new Set(previous.filter(j => j.status === 'running').map(j => j.job_id))

  for (const job of next) {
    if (
      job.status === 'running' ||
      (!wasRunning.has(job.job_id) && !acceptedIds.has(job.job_id)) ||
      settledNotified.has(job.job_id)
    ) {
      continue
    }

    settledNotified.add(job.job_id)
    acceptedIds.delete(job.job_id)

    if (job.status === 'done') {
      notify({
        durationMs: 6_000,
        kind: 'success',
        title: translateNow('settings.localModels.title'),
        message:
          job.kind === 'model-download'
            ? translateNow('settings.localModels.downloadDoneToast', job.target)
            : job.kind === 'model-activate'
              ? translateNow('settings.localModels.activateDoneToast', job.target)
              : job.kind === 'quickstart'
                ? translateNow('settings.localModels.quickstartDoneToast', job.target)
                : translateNow('settings.localModels.installDoneToast')
      })
    } else {
      notifyError(
        new Error(job.error ?? job.detail ?? 'failed'),
        job.kind === 'model-download'
          ? translateNow('settings.localModels.downloadFailed', job.target)
          : job.kind === 'model-activate'
            ? translateNow('settings.localModels.activateFailed', job.target)
            : job.kind === 'quickstart'
              ? translateNow('settings.localModels.quickstartFailed')
              : translateNow('settings.localModels.installFailed')
      )
    }
  }
}

function poll(): Promise<void> {
  if (polling) {
    return polling
  }

  const context = generation
  const owner = activeContext
  const accepted = owner.acceptedInstall

  if (timer !== null) {
    clearTimeout(timer)
  }

  timer = null
  polling = (async () => {
    try {
      const { jobs } = await getLocalModelsJobs()

      if (context !== generation || accepted !== owner.acceptedInstall) {
        return
      }

      const previous = $localRuntimeJobs.get()

      // Acceptance can arrive after a pane already read the terminal job.
      notifySettled(previous, jobs)

      if (!jobsEqual(previous, jobs)) {
        $localRuntimeJobs.set(jobs)
      }

      owner.jobs = jobs

      if (owner.readPending) {
        owner.readPending = false
        $localRuntimeInstallStarting.set(owner.postPending)
      }
    } catch {
      // Backend unreachable — keep the last snapshot; the next poll retries.
    } finally {
      if (context === generation) {
        polling = null

        if (owner.readPending || $localRuntimeJobs.get().some(j => j.status === 'running')) {
          timer = setTimeout(() => void poll(), POLL_ACTIVE_MS)
        }
      }
    }
  })()

  return polling
}

// Idempotent kick: start (or keep) the poll loop while work is in flight.
// Call after starting a job AND on app boot (to rediscover work started
// before a reload).
export function watchLocalRuntimeJobs() {
  if (polling || timer !== null) {
    return
  }

  void poll()
}

// Selector: the running download job for a catalog model id, if any.
export function runningDownloadFor(jobs: readonly LocalRuntimeJob[], modelId: string): LocalRuntimeJob | null {
  return jobs.find(j => j.kind === 'model-download' && j.status === 'running' && j.model_id === modelId) ?? null
}

// Selector: every model on its way to the library right now — plain
// downloads plus quickstart runs while they are still fetching bytes
// (later quickstart phases mean the model is staged and activating).
// The model picker renders these as disabled progress rows.
const DOWNLOAD_PHASES = new Set(['starting', 'installing-runtime', 'downloading'])

export function runningModelDownloads(jobs: readonly LocalRuntimeJob[]): LocalRuntimeJob[] {
  return jobs.filter(
    j =>
      j.status === 'running' &&
      (j.kind === 'model-download' || (j.kind === 'quickstart' && DOWNLOAD_PHASES.has(j.phase)))
  )
}

export function runningRuntimeInstall(jobs: readonly LocalRuntimeJob[]): LocalRuntimeJob | null {
  return jobs.find(j => j.kind === 'runtime-install' && j.status === 'running') ?? null
}
