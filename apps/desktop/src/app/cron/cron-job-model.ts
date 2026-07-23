import type { CronJob, CronJobUpdates, CronReasoningEffort } from '@/types/hermes'

/** Hermes' canonical per-job reasoning choices, plus `none` to disable thinking. */
export const CRON_REASONING_EFFORT_VALUES = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
  'ultra'
] as const

export function isCronReasoningEffort(value: string): value is CronReasoningEffort {
  return (CRON_REASONING_EFFORT_VALUES as readonly string[]).includes(value)
}

export interface CronReasoningEffortFormState {
  formValue: string
  preserveOnSave: boolean
}

export function classifyCronReasoningEffort(job: Pick<CronJob, 'reasoning_effort'>): CronReasoningEffortFormState {
  if (!Object.prototype.hasOwnProperty.call(job, 'reasoning_effort')) {
    return { formValue: '', preserveOnSave: false }
  }

  const rawValue = job.reasoning_effort

  if (rawValue === null) {
    return { formValue: '', preserveOnSave: false }
  }

  if (rawValue === false) {
    return { formValue: 'none', preserveOnSave: false }
  }

  if (typeof rawValue === 'string' && isCronReasoningEffort(rawValue)) {
    return { formValue: rawValue, preserveOnSave: false }
  }

  return { formValue: typeof rawValue === 'string' ? rawValue : '', preserveOnSave: true }
}

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
  /** Per-job reasoning override (''/null = clear and inherit; undefined = preserve an unknown legacy value). */
  reasoningEffort?: CronReasoningEffort | '' | null
  /** Whether the user changed the reasoning picker during this edit. */
  reasoningChanged?: boolean
  /** Keep an invalid legacy value when no reasoning change was made. */
  reasoningPreserveOnSave?: boolean
  schedule: string
}

/** Build the API update payload, preserving empty prompts and no-agent settings. */
export function cronEditorUpdates(
  values: CronEditorSaveValues,
  options: { noAgentJob?: boolean; scriptOnlyJob: boolean }
): CronJobUpdates {
  const updates: CronJobUpdates = {
    deliver: values.deliver,
    name: values.name,
    schedule: values.schedule.trim()
  }

  const trimmedPrompt = values.prompt.trim()

  if (!options.scriptOnlyJob || trimmedPrompt) {
    updates.prompt = trimmedPrompt
  }

  // Script-only jobs never run an agent, so leave their model/provider settings
  // untouched. A no-agent job also preserves its configured reasoning effort,
  // while agent jobs write all override axes so inherit clears previous pins.
  if (!options.scriptOnlyJob) {
    updates.model = values.model.trim() || null
    updates.provider = values.provider.trim() || null

    if (
      !options.noAgentJob &&
      (!values.reasoningPreserveOnSave || values.reasoningChanged) &&
      values.reasoningEffort !== undefined
    ) {
      if (values.reasoningEffort === null || values.reasoningEffort === '') {
        updates.reasoning_effort = null
      } else if (isCronReasoningEffort(values.reasoningEffort)) {
        updates.reasoning_effort = values.reasoningEffort
      }
    }
  }

  return updates
}
