import { type CronJob, getCronJobRuns, type SessionInfo } from '@/hermes'
import { translateNow } from '@/i18n'

import { dispatchNativeNotification, type NativeNotificationInput } from './native-notifications'

const FAILED_RUN_STATUSES = new Set(['error', 'failed', 'timeout'])
const MAX_RUN_LOOKUP_MISSES = 3

interface CronCompletionNotifierDependencies {
  getRuns: (jobId: string, limit?: number) => Promise<SessionInfo[]>
  notify: (input: NativeNotificationInput) => boolean | void
}

export interface CronCompletionNotifier {
  observe: (scope: string, jobs: CronJob[]) => Promise<void>
  reset: () => void
}

function deliversLocally(job: CronJob): boolean {
  const targets = (job.deliver ?? 'local')
    .split(',')
    .map(target => target.trim())
    .filter(Boolean)

  return targets.length === 0 || targets.includes('local')
}

function runFailed(job: CronJob): boolean {
  const status = job.last_status?.trim().toLowerCase()

  return Boolean(job.last_error) || Boolean(status && FAILED_RUN_STATUSES.has(status))
}

function runLookupKey(jobId: string, completedAt: string): string {
  return `${jobId}\u0000${completedAt}`
}

function runNearestCompletion(runs: SessionInfo[], completedAt: string): SessionInfo | undefined {
  const completedMs = Date.parse(completedAt)

  if (Number.isNaN(completedMs)) {
    return runs[0]
  }

  let nearest: SessionInfo | undefined
  let nearestDistance = Number.POSITIVE_INFINITY

  for (const run of runs) {
    const runSeconds = run.ended_at ?? run.last_active ?? run.started_at
    const distance = Math.abs(runSeconds * 1000 - completedMs)

    if (distance < nearestDistance) {
      nearest = run
      nearestDistance = distance
    }
  }

  return nearest ?? runs[0]
}

function advanceSeenRun(seenRuns: Map<string, null | string>, jobId: string, completedAt: string): void {
  const current = seenRuns.get(jobId)

  if (current) {
    const currentMs = Date.parse(current)
    const completedMs = Date.parse(completedAt)

    if (!Number.isNaN(currentMs) && !Number.isNaN(completedMs) && completedMs < currentMs) {
      return
    }
  }

  seenRuns.set(jobId, completedAt)
}

export function createCronCompletionNotifier({
  getRuns,
  notify
}: CronCompletionNotifierDependencies): CronCompletionNotifier {
  let activeScope: null | string = null
  let generation = 0
  const seenRuns = new Map<string, null | string>()
  const inFlight = new Set<string>()
  const runLookupMisses = new Map<string, number>()

  const reset = () => {
    activeScope = null
    generation += 1
    seenRuns.clear()
    inFlight.clear()
    runLookupMisses.clear()
  }

  const observe = async (scope: string, jobs: CronJob[]) => {
    if (scope !== activeScope) {
      activeScope = scope
      generation += 1
      seenRuns.clear()
      inFlight.clear()
      runLookupMisses.clear()

      for (const job of jobs) {
        seenRuns.set(job.id, job.last_run_at ?? null)
      }

      return
    }

    const liveJobIds = new Set(jobs.map(job => job.id))

    for (const jobId of seenRuns.keys()) {
      if (!liveJobIds.has(jobId)) {
        seenRuns.delete(jobId)
      }
    }

    for (const lookupKey of runLookupMisses.keys()) {
      const jobId = lookupKey.split('\u0000', 1)[0]

      if (!liveJobIds.has(jobId)) {
        runLookupMisses.delete(lookupKey)
      }
    }

    const observedGeneration = generation

    const candidates = jobs.flatMap(job => {
      const completedAt = job.last_run_at ?? null

      if (!seenRuns.has(job.id)) {
        seenRuns.set(job.id, completedAt)

        return []
      }

      if (seenRuns.get(job.id) === completedAt) {
        return []
      }

      if (!completedAt || !deliversLocally(job)) {
        seenRuns.set(job.id, completedAt)

        return []
      }

      const key = `${observedGeneration}\u0000${job.id}\u0000${completedAt}`

      if (inFlight.has(key)) {
        return []
      }

      inFlight.add(key)

      return [{ completedAt, job, key }]
    })

    await Promise.all(
      candidates.map(async ({ completedAt, job, key }) => {
        try {
          const runs = await getRuns(job.id, 5)
          const run = runNearestCompletion(runs, completedAt)

          if (generation !== observedGeneration || activeScope !== scope) {
            return
          }

          const lookupKey = runLookupKey(job.id, completedAt)

          if (!run) {
            const missCount = (runLookupMisses.get(lookupKey) ?? 0) + 1

            if (missCount >= MAX_RUN_LOOKUP_MISSES) {
              advanceSeenRun(seenRuns, job.id, completedAt)
              runLookupMisses.delete(lookupKey)
            } else {
              runLookupMisses.set(lookupKey, missCount)
            }

            return
          }

          runLookupMisses.delete(lookupKey)

          // Two accepted snapshots can advance the same job while the first
          // run lookup is still in flight. Never let the older lookup resolve
          // last and move the dedupe watermark backwards.
          advanceSeenRun(seenRuns, job.id, completedAt)
          const failed = runFailed(job)

          notify({
            body: job.last_error || job.name || job.id,
            global: true,
            kind: 'backgroundDone',
            sessionId: run.id,
            tag: `cron:${job.id}:${completedAt}`,
            title: translateNow(
              failed
                ? 'notifications.native.backgroundFailedTitle'
                : 'notifications.native.backgroundDoneTitle'
            )
          })
        } catch {
          if (generation !== observedGeneration || activeScope !== scope) {
            return
          }

          const lookupKey = runLookupKey(job.id, completedAt)
          const missCount = (runLookupMisses.get(lookupKey) ?? 0) + 1

          if (missCount >= MAX_RUN_LOOKUP_MISSES) {
            advanceSeenRun(seenRuns, job.id, completedAt)
            runLookupMisses.delete(lookupKey)
          } else {
            runLookupMisses.set(lookupKey, missCount)
          }

          // The jobs file can become visible just before its session row. Keep
          // the timestamp unconsumed for a bounded retry window, then consume it
          // silently so a removed or unavailable runs endpoint cannot hot-loop
          // forever on every cron refresh.
        } finally {
          inFlight.delete(key)
        }
      })
    )
  }

  return { observe, reset }
}

const cronCompletionNotifier = createCronCompletionNotifier({
  getRuns: getCronJobRuns,
  notify: dispatchNativeNotification
})

export function observeCronCompletions(scope: string, jobs: CronJob[]): Promise<void> {
  return cronCompletionNotifier.observe(scope, jobs)
}

export function resetCronCompletionNotifications(): void {
  cronCompletionNotifier.reset()
}
