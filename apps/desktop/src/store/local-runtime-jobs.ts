import { atom } from 'nanostores'

import { getLocalModelsJobs, getLocalModelsStatus } from '@/hermes'
import { translateNow } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import type { LocalRuntimeJob } from '@/types/hermes'

// App-level tracker for local-runtime jobs (runtime installs, model
// downloads). The AUTHORITY is the backend job registry — this store is a
// cache of it (desktop guide: server truth is cached, not owned). Living at
// the store layer, not in the settings pane, is what makes a download
// survive the pane unmounting: anything can start a job, the poller follows
// it to completion, and completion/failure notify app-wide exactly once.

export const $localRuntimeJobs = atom<readonly LocalRuntimeJob[]>([])

const POLL_ACTIVE_MS = 700
let timer: null | number = null
let polling = false
// Jobs we've already toasted for, so a poll race can't double-notify.
const settledNotified = new Set<string>()

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
  const wasRunning = new Set(previous.filter(j => j.status === 'running').map(j => j.job_id))

  for (const job of next) {
    if (job.status === 'running' || !wasRunning.has(job.job_id) || settledNotified.has(job.job_id)) {
      continue
    }

    settledNotified.add(job.job_id)

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

async function poll() {
  try {
    const { jobs } = await getLocalModelsJobs()
    const previous = $localRuntimeJobs.get()

    if (!jobsEqual(previous, jobs)) {
      notifySettled(previous, jobs)
      $localRuntimeJobs.set(jobs)
    }
  } catch {
    // Backend unreachable — keep the last snapshot; the next poll retries.
  }

  const anyRunning = $localRuntimeJobs.get().some(j => j.status === 'running')

  if (anyRunning) {
    timer = window.setTimeout(() => void poll(), POLL_ACTIVE_MS)
  } else {
    polling = false
    timer = null
  }
}

// Idempotent kick: start (or keep) the poll loop while work is in flight.
// Call after starting a job AND on app boot (to rediscover work started
// before a reload).
export function watchLocalRuntimeJobs() {
  if (polling) {
    return
  }

  polling = true

  if (timer !== null) {
    window.clearTimeout(timer)
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

// One engine-update toast per app session: checked at boot (after the
// gateway is ready), only when the user runs the local engine. The
// download itself is always a button click in Local Models — this is a
// pointer, not an installer.
let updateNotified = false

export async function checkLocalRuntimeUpdate() {
  if (updateNotified) {
    return
  }

  try {
    const status = await getLocalModelsStatus()

    if (status.enabled && status.update_available) {
      updateNotified = true
      notify({
        durationMs: 10_000,
        kind: 'info',
        title: translateNow('settings.localModels.title'),
        message: translateNow('settings.localModels.updateToast', status.configured_tag)
      })
    }
  } catch {
    // Backend without the endpoint (older runtime) or transient failure —
    // silently skip; the pane still shows the update row when opened.
  }
}
