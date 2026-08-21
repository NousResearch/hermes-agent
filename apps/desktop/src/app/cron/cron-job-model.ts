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
  /** Per-job model override ('' = follow the cron/global default at fire time). */
  model: string
  name: string
  prompt: string
  /** Provider for the model override ('' = none). Always paired with model. */
  provider: string
  schedule: string
}

/**
 * Routing status for an agent job's inference model, mirroring the backend's
 * fire-time resolution (per-job pin > cron.model > global model.default).
 * Drift-guard visibility (#89513): a job with no explicit pin may resolve to
 * the cron-fleet default (cron.model) rather than the global chat model.
 */
export type CronModelRoutingKind = 'pinned' | 'fleet' | 'global'

export interface CronFleetConfig {
  model: string
  provider: string
}

export function jobModelRouting(
  job: Pick<CronJob, 'model' | 'provider'>,
  fleet: CronFleetConfig | null
): { kind: CronModelRoutingKind; label: string } {
  const model = String(job.model ?? '').trim()
  const provider = String(job.provider ?? '').trim()

  if (model) {
    return { kind: 'pinned', label: provider ? `${provider} · ${model}` : model }
  }

  const fleetModel = fleet?.model?.trim()

  if (fleetModel) {
    const fleetProvider = fleet?.provider?.trim()

    return { kind: 'fleet', label: fleetProvider ? `${fleetProvider} · ${fleetModel}` : fleetModel }
  }

  return { kind: 'global', label: 'Global default' }
}

/** Sentinel for the cron-fleet default (cron.model) in the model picker. */
export const MODEL_FLEET_VALUE = '__fleet__'

export function cronModelChoiceLabel(choice: string, fleet: CronFleetConfig | null): string {
  if (choice === MODEL_FLEET_VALUE) {
    const fleetLabel = fleet?.model?.trim() || ''
    const fleetProvider = fleet?.provider?.trim()
    const rendered = fleetProvider && fleetLabel ? `${fleetProvider} · ${fleetLabel}` : fleetLabel

    return rendered ? `Fleet default (cron.model): ${rendered}` : 'Fleet default (cron.model)'
  }

  return 'Default (global model)'
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
