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
}

/** Build the narrow update payload used by the detail-panel model picker. */
export function cronModelUpdates(provider: string, model: string): CronJobUpdates {
  return {
    model: model.trim() || null,
    provider: provider.trim() || null
  }
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

/** Build the API update payload, preserving an empty prompt on script-only jobs. */
export function cronEditorUpdates(values: CronEditorSaveValues, options: { scriptOnlyJob: boolean }): CronJobUpdates {
  const updates: CronJobUpdates = {
    deliver: values.deliver,
    name: values.name,
    schedule: values.schedule.trim()
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
