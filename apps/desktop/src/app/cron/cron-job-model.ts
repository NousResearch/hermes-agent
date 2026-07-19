import type { CronJob, CronJobCreatePayload, CronJobUpdates, ModelOptionProvider } from '@/types/hermes'

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Sentinel Select value for "follow the profile's current default model". */
export const CRON_MODEL_DEFAULT = '__default__'

const MODEL_CHOICE_SEP = '::'

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

export interface CronModelChoice {
  model: null | string
  provider: null | string
}

/** Encode a pinned provider/model pair for a single Select value. */
export function encodeCronModelChoice(provider: string, model: string): string {
  const trimmedProvider = provider.trim()
  const trimmedModel = model.trim()

  if (!trimmedProvider || !trimmedModel) {
    return CRON_MODEL_DEFAULT
  }

  return `${trimmedProvider}${MODEL_CHOICE_SEP}${trimmedModel}`
}

/** Decode a Select value back into optional provider/model pins. */
export function decodeCronModelChoice(value: string): CronModelChoice {
  const trimmed = value.trim()

  if (!trimmed || trimmed === CRON_MODEL_DEFAULT) {
    return { model: null, provider: null }
  }

  const separator = trimmed.indexOf(MODEL_CHOICE_SEP)

  if (separator <= 0) {
    return { model: null, provider: null }
  }

  const provider = trimmed.slice(0, separator).trim()
  const model = trimmed.slice(separator + MODEL_CHOICE_SEP.length).trim()

  if (!provider || !model) {
    return { model: null, provider: null }
  }

  return { model, provider }
}

/** Select value for an existing job (default when unpinned). */
export function cronModelSelectValue(job: Pick<CronJob, 'model' | 'provider'>): string {
  return encodeCronModelChoice(asText(job.provider), asText(job.model))
}

export interface CronModelOption {
  label: string
  model: string
  provider: string
  value: string
}

/** Flatten authenticated provider catalogs into Select options. */
export function buildCronModelOptions(providers: ModelOptionProvider[] | undefined): CronModelOption[] {
  if (!providers?.length) {
    return []
  }

  const options: CronModelOption[] = []
  const seen = new Set<string>()

  for (const row of providers) {
    if (row.authenticated === false) {
      continue
    }

    const provider = (row.slug || row.name || '').trim()

    if (!provider) {
      continue
    }

    const providerLabel = (row.name || row.slug || provider).trim()
    const unavailable = new Set(row.unavailable_models ?? [])

    for (const model of row.models ?? []) {
      const trimmedModel = model.trim()

      if (!trimmedModel || unavailable.has(trimmedModel)) {
        continue
      }

      const value = encodeCronModelChoice(provider, trimmedModel)

      if (seen.has(value)) {
        continue
      }

      seen.add(value)
      options.push({
        label: `${providerLabel} · ${trimmedModel}`,
        model: trimmedModel,
        provider,
        value
      })
    }
  }

  return options
}

/**
 * Keep a pinned job selection visible even when it left the live catalog
 * (stale pin / provider temporarily empty).
 */
export function ensureCronModelOption(
  options: CronModelOption[],
  choice: CronModelChoice
): CronModelOption[] {
  const provider = choice.provider?.trim() ?? ''
  const model = choice.model?.trim() ?? ''

  if (!provider || !model) {
    return options
  }

  const value = encodeCronModelChoice(provider, model)

  if (options.some(option => option.value === value)) {
    return options
  }

  return [{ label: `${provider} · ${model}`, model, provider, value }, ...options]
}

export interface CronEditorSaveValues {
  deliver: string
  model: null | string
  name: string
  prompt: string
  provider: null | string
  schedule: string
}

/** Build the create payload; omit pins when following the default model. */
export function cronEditorCreatePayload(values: CronEditorSaveValues): CronJobCreatePayload {
  const payload: CronJobCreatePayload = {
    deliver: values.deliver,
    prompt: values.prompt.trim(),
    schedule: values.schedule.trim()
  }

  const name = values.name.trim()

  if (name) {
    payload.name = name
  }

  const model = values.model?.trim() || null
  const provider = values.provider?.trim() || null

  if (model) {
    payload.model = model

    if (provider) {
      payload.provider = provider
    }
  }

  return payload
}

/** Build the API update payload, preserving an empty prompt on script-only jobs. */
export function cronEditorUpdates(values: CronEditorSaveValues, options: { scriptOnlyJob: boolean }): CronJobUpdates {
  const updates: CronJobUpdates = {
    deliver: values.deliver,
    name: values.name,
    schedule: values.schedule.trim()
  }

  // Script-only edits hide the Model control — omit inference pins so a save
  // cannot wipe a previously stored provider/model pair.
  if (!options.scriptOnlyJob) {
    updates.model = values.model?.trim() || null
    updates.provider = values.provider?.trim() || null
  }

  const trimmedPrompt = values.prompt.trim()

  if (!options.scriptOnlyJob || trimmedPrompt) {
    updates.prompt = trimmedPrompt
  }

  return updates
}
