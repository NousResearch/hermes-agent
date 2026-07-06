import { computed } from 'nanostores'

import { translateNow } from '@/i18n'
import type { CronJob } from '@/types/hermes'

import { isOneShotJob, jobTitleShort, type TaskPhase, taskPhase } from '@/app/tasks/task-model'

import { $cronJobs } from './cron'
import { dispatchNativeNotification } from './native-notifications'

// The tasks page (long-running one-shot "Goal-mode" runs) is a projection over
// the same cron-jobs atom the controller already polls — no independent fetch.
// A task == a one-shot cron job (see task-model.ts for the honest rationale).
export const $tasks = computed($cronJobs, jobs => jobs.filter(isOneShotJob))

// ── Completion / failure notifications ──────────────────────────────────────
// The cron ticker advances job state in the backend; the controller poll swaps
// $cronJobs. We watch phase transitions here and fire a native OS notification
// when a task leaves the running phase, so the user learns a background job
// finished (or failed) without staring at the page. Unlike the composer's
// per-session backgroundDone (gated to the active session), a task can finish
// while the user is anywhere, so this fires regardless of the active view — the
// whole point is "kick it off and walk away".

const lastPhase = new Map<string, TaskPhase>()
// Guard so we don't fire on the very first snapshot after launch (every existing
// finished task would alert at once). We only notify on a transition we observed.
let seeded = false

function notifyTransition(job: CronJob, phase: TaskPhase): void {
  const title = jobTitleShort(job)

  if (phase === 'done') {
    dispatchNativeNotification({
      kind: 'backgroundDone',
      title: translateNow('tasks.notify.doneTitle'),
      body: title
    })
  } else if (phase === 'failed') {
    dispatchNativeNotification({
      kind: 'turnError',
      title: translateNow('tasks.notify.failedTitle'),
      body: job.last_error?.trim() || title
    })
  }
}

/** Subscribe once (from the controller) so task completion/failure alerts fire
 *  for the whole app session. Returns an unsubscribe for symmetry/tests. */
export function startTaskNotifier(): () => void {
  return $tasks.subscribe(tasks => {
    const seen = new Set<string>()

    for (const job of tasks) {
      seen.add(job.id)
      const phase = taskPhase(job)
      const prev = lastPhase.get(job.id)
      lastPhase.set(job.id, phase)

      // Fire only on a real running → terminal edge that we've been tracking.
      if (seeded && prev === 'running' && (phase === 'done' || phase === 'failed')) {
        notifyTransition(job, phase)
      }
    }

    // Drop tasks that disappeared (deleted) so the map can't grow unbounded.
    for (const id of [...lastPhase.keys()]) {
      if (!seen.has(id)) {
        lastPhase.delete(id)
      }
    }

    seeded = true
  })
}

// Test seam: reset the module's transition memory between cases.
export function __resetTaskNotifierState(): void {
  lastPhase.clear()
  seeded = false
}
