import type { CronJob, CronJobUpdates, CronReasoningEffortOption } from '@/types/hermes'

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

// The backend remains the final validator. This tuple is the renderer's
// closed picker grammar, including its two UI sentinels.
export const CRON_REASONING_VALUES = [
  'inherit',
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
  'ultra'
] as const satisfies readonly CronReasoningEffortOption[]

export type CronReasoningValue = CronReasoningEffortOption

const CRON_REASONING_VALUE_SET: ReadonlySet<string> = new Set(CRON_REASONING_VALUES)

/** Clamp stored or UI input to the closed picker grammar. */
export function normalizeCronReasoningEffort(value: unknown): CronReasoningValue {
  // YAML/JSON boolean false is the backend's explicit disable spelling.
  if (value === false) {
    return 'none'
  }

  if (typeof value !== 'string') {
    return 'inherit'
  }

  const normalized = value.trim().toLowerCase()

  return CRON_REASONING_VALUE_SET.has(normalized) ? (normalized as CronReasoningValue) : 'inherit'
}

/** Map the picker grammar onto the backend's nullable override field. */
export function cronReasoningEffortToWire(value: unknown): null | string {
  const normalized = normalizeCronReasoningEffort(value)

  return normalized === 'inherit' ? null : normalized
}

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
  /** Per-job reasoning effort ('inherit' = follow config resolution at fire time). */
  reasoningEffort: CronReasoningValue
  /** Provider for the model override ('' = none). Always paired with model. */
  provider: string
  schedule: string
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

  // Script-only jobs never run an agent, so the scheduler ignores model and
  // reasoning overrides — leave whatever is stored untouched. For agent
  // jobs, always write all three axes so resetting to "default" clears a
  // previous pin (the backend normalizes null/'' to "no override").
  if (!options.scriptOnlyJob) {
    updates.model = values.model.trim() || null
    updates.provider = values.provider.trim() || null
    updates.reasoning_effort = cronReasoningEffortToWire(values.reasoningEffort)
  }

  return updates
}
