import type { CronJob } from '@/hermes'

import { lastErrorSummary } from '../cron/cron-job-model'

import { routineIssue } from './routine-model'
import type { RoutinesCopy } from './routines-copy'

// The scheduler refuses a run when the routine's agent has no gateway serving it here (a parked
// profile, or one this machine's gateway does not serve). The raw text names CLI migrations; the
// person reading the routine only needs to know its agent is not running.
const AGENT_NOT_RUNNING_RE = /\bprofile_(?:parked|unserved|unavailable)\b|\bdoes not serve (?:agent|profile)\b/i

/** The backend's full text of what went wrong with the routine's last run, or '' when nothing did. */
export function routineErrorText(job: CronJob): string {
  const issue = routineIssue(job)
  const raw = issue === 'failed' ? (job.latest_execution?.error ?? job.last_error) : issue === 'delivery' ? job.last_delivery_error : ''

  return (raw ?? '').trim()
}

/** One plain sentence for a run's error: a known refusal in the routine's words, otherwise the
 *  backend message's first sentence with its wrappers peeled. */
export function plainRunError(raw: string, copy: RoutinesCopy, agent: string): string {
  return AGENT_NOT_RUNNING_RE.test(raw) ? copy.banner.agentNotRunning(agent) : lastErrorSummary(raw)
}
