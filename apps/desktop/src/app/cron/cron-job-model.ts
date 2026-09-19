import type { CronJob, CronJobUpdates } from '@/types/hermes'

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Script-only cron jobs run a shell script on schedule with no LLM prompt. */
export function jobIsScriptOnly(job: Pick<CronJob, 'no_agent' | 'script'>): boolean {
  return Boolean(job.no_agent) && Boolean(asText(job.script).trim())
}

export type CronEditorValidationError = 'prompt' | 'prompt_and_schedule' | 'schedule'

export interface CronEditorValidationInput {
  prompt: string
  schedule: string
  scriptOnlyJob: boolean
}

export function validateCronEditor(input: CronEditorValidationInput): CronEditorValidationError | null {
  const trimmedPrompt = input.prompt.trim()
  const trimmedSchedule = input.schedule.trim()

  if (!trimmedSchedule && !trimmedPrompt && !input.scriptOnlyJob) {
    return 'prompt_and_schedule'
  }

  if (!trimmedSchedule) {
    return 'schedule'
  }

  if (!input.scriptOnlyJob && !trimmedPrompt) {
    return 'prompt'
  }

  return null
}

export interface CronEditorSaveValues {
  deliver: string
  /** Per-job model override ('' = follow the global default at fire time). */
  model: string
  name: string
  prompt: string
  /** Provider for the model override ('' = none). Always paired with model. */
  provider: string
  schedule: string
  no_agent?: boolean
  script?: string
}

export function parseCronDeliveryTargets(value: string): string[] {
  const targets = value
    .split(',')
    .map(target => target.trim())
    .filter(Boolean)

  return targets.length > 0 ? [...new Set(targets)] : ['local']
}

export function toggleCronDeliveryTarget(value: string, target: string, checked: boolean): string {
  const targets = parseCronDeliveryTargets(value)

  if (checked) {
    return targets.includes(target) ? targets.join(',') : [...targets, target].join(',')
  }

  if (!targets.includes(target) || targets.length === 1) {
    return targets.join(',')
  }

  return targets.filter(candidate => candidate !== target).join(',')
}

// The scheduler stores `last_error` as the raw exception text, e.g.
// "RuntimeError: Cron job 'x' has no model configured (job.model=None, …). Set a
// per-job model via `hermes cron edit …`". Users need the first plain sentence,
// not the Python wrapper; the full text stays reachable via a hover title.
const ERROR_PREFIX_RE = /^(?:[A-Za-z_][\w.]*(?:Error|Exception)|Exception):\s*/
const ERROR_MARKER_RE = /^\[[a-z_]+(?::[a-z_]+)?\]\s*/
const ERROR_EMOJI_RE = /^(?:\u26A0\uFE0F?|\uD83D\uDED1|\u274C|\u{1F6AB})\s*/u
const ERROR_SUMMARY_MAX = 200

export function lastErrorSummary(lastError: string | null | undefined): string {
  let text = (lastError ?? '').trim()

  // Wrappers can nest (marker, then emoji, then exception class); peel until stable.
  for (let previous = ''; previous !== text; ) {
    previous = text
    text = text.replace(ERROR_MARKER_RE, '').replace(ERROR_EMOJI_RE, '').replace(ERROR_PREFIX_RE, '').trimStart()
  }

  const sentenceEnd = text.search(/\. |\n/)
  const sentence = (sentenceEnd === -1 ? text : text.slice(0, sentenceEnd + 1)).trim()

  return sentence.length > ERROR_SUMMARY_MAX ? `${sentence.slice(0, ERROR_SUMMARY_MAX - 1).trimEnd()}…` : sentence
}

/** Build the API update payload, preserving an empty prompt on script-only jobs. */
export function cronEditorUpdates(values: CronEditorSaveValues, options: { scriptOnlyJob: boolean }): CronJobUpdates {
  const updates: CronJobUpdates = {
    deliver: values.deliver,
    name: values.name,
    schedule: values.schedule.trim()
  }

  if (options.scriptOnlyJob) {
    updates.no_agent = true
    const trimmedScript = values.script?.trim()
    if (trimmedScript) {
      updates.script = trimmedScript
    }
  }

  const trimmedPrompt = values.prompt.trim()

  if (!options.scriptOnlyJob || trimmedPrompt) {
    updates.prompt = trimmedPrompt
  }

  // Script-only jobs never run an agent, so the scheduler ignores model
  // overrides — leave whatever is stored untouched. For agent jobs, always
  // write both axes so resetting to "default" clears a previous pin (the
  // backend normalizes null/'' to "no override").
  if (!options.scriptOnlyJob) {
    updates.model = values.model.trim() || null
    updates.provider = values.provider.trim() || null
  }

  return updates
}
