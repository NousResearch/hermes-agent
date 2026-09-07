import type { CronJob, CronJobUpdates } from '@/types/hermes'

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Script-only cron jobs run a shell script on schedule with no LLM prompt. */
export function jobIsScriptOnly(job: Pick<CronJob, 'no_agent' | 'script'>): boolean {
  return Boolean(job.no_agent) && Boolean(asText(job.script).trim())
}

export type CronEditorValidationError = 'monitor' | 'prompt' | 'prompt_and_schedule' | 'repeat' | 'schedule'

export interface CronEditorValidationInput {
  advanced?: CronAdvancedValues
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

  const advanced = input.advanced ?? emptyCronAdvancedValues()

  if (advanced.monitorScript.trim() && advanced.monitorUrl.trim()) {
    return 'monitor'
  }

  if (Number.isNaN(parseRepeatTimes(advanced.repeat))) {
    return 'repeat'
  }

  return null
}

export interface CronEditorSaveValues {
  advanced?: CronAdvancedValues
  deliver: string
  /** Per-job model override ('' = follow the global default at fire time). */
  model: string
  name: string
  prompt: string
  /** Provider for the model override ('' = none). Always paired with model. */
  provider: string
  schedule: string
}

/** Advanced execution settings — the collapsed section. Text-first shapes keep
 *  the form serializable; the updates builder converts to backend shapes. */
export interface CronAdvancedValues {
  /** Explicit session attach (checkbox). */
  attachToSession: boolean
  /** Job ids whose latest output chains into the prompt (comma-separated). */
  contextFrom: string
  /** Toolsets enabled for the run (comma-separated). */
  enabledToolsets: string
  /** Failure delivery override ('' = fall back to deliver). */
  failureDeliver: string
  /** Monitor source run first each tick; output change wakes the agent. */
  monitorScript: string
  /** ...or a URL monitor source (mutually exclusive with monitorScript). */
  monitorUrl: string
  /** Per-job reasoning pin ('' = none). Validated against the shared parser. */
  reasoningEffort: string
  /** Repeat count ('' = forever). */
  repeat: string
  /** Skills for the run (comma-separated). */
  skills: string
  /** Absolute cwd for tools/scripts. */
  workdir: string
}

export function emptyCronAdvancedValues(): CronAdvancedValues {
  return {
    attachToSession: false,
    contextFrom: '',
    enabledToolsets: '',
    failureDeliver: '',
    monitorScript: '',
    monitorUrl: '',
    reasoningEffort: '',
    repeat: '',
    skills: '',
    workdir: ''
  }
}

/** Split a comma-separated form field into a clean list. */
export function parseCommaList(value: string): string[] {
  return [...new Set(
    value.split(',').map(part => part.trim()).filter(Boolean)
  )]
}

/** Stored repeat (bare number, {times, completed}, or null) → form text. */
export function formatRepeatTimes(repeat: CronJob['repeat']): string {
  if (repeat === null || repeat === undefined) {
    return ''
  }

  const times = typeof repeat === 'number' ? repeat : repeat.times

  return typeof times === 'number' && times > 0 ? String(times) : ''
}

/** Seed the advanced form from a stored job (edit mode). */
export function seedCronAdvancedValues(job: CronJob): CronAdvancedValues {
  const list = (value: unknown): string =>
    Array.isArray(value)
      ? value.filter(item => typeof item === 'string' && item.trim()).join(', ')
      : typeof value === 'string' ? value : ''

  return {
    attachToSession: job.attach_to_session === true,
    contextFrom: list(job.context_from),
    enabledToolsets: list(job.enabled_toolsets),
    failureDeliver: typeof job.failure_deliver === 'string' ? job.failure_deliver : '',
    monitorScript: typeof job.monitor_script === 'string' ? job.monitor_script : '',
    monitorUrl: typeof job.monitor_url === 'string' ? job.monitor_url : '',
    reasoningEffort: typeof job.reasoning_effort === 'string' ? job.reasoning_effort : '',
    repeat: formatRepeatTimes(job.repeat),
    skills: list(job.skills),
    workdir: typeof job.workdir === 'string' ? job.workdir : ''
  }
}

/** Parse a repeat form value: '' → null (forever), digits → int, else NaN. */
export function parseRepeatTimes(value: string): null | number {
  const trimmed = value.trim()

  if (!trimmed) {
    return null
  }

  if (!/^\d+$/.test(trimmed)) {
    return Number.NaN
  }

  const times = Number.parseInt(trimmed, 10)

  return times > 0 ? times : null
}

/** Stable summary keys for the detail view's effective-settings rows. */
export type CronAdvancedSummaryKey =
  | 'attachToSession'
  | 'contextFrom'
  | 'enabledToolsets'
  | 'executionMode'
  | 'failureDeliver'
  | 'monitor'
  | 'reasoningEffort'
  | 'repeat'
  | 'skills'
  | 'workdir'

/** Read-only effective-settings rows for the job detail view (issue point 5):
 *  only fields the backend actually has set. Execution mode derives from the
 *  stored mode fields (script-only / monitor / agent) so the row is honest
 *  even for jobs created via CLI/chat. */
export function cronAdvancedSummaryRows(job: CronJob): { key: CronAdvancedSummaryKey; value: string }[] {
  const rows: { key: CronAdvancedSummaryKey; value: string }[] = []
  const text = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

  const list = (value: unknown): string =>
    Array.isArray(value)
      ? value.filter(item => typeof item === 'string' && item.trim()).join(', ')
      : text(value)

  if (jobIsScriptOnly(job)) {
    rows.push({ key: 'executionMode', value: 'script' })
  } else if (text(job.monitor_script) || text(job.monitor_url)) {
    rows.push({ key: 'executionMode', value: 'monitor' })
  } else {
    rows.push({ key: 'executionMode', value: 'agent' })
  }

  const repeatText = formatRepeatTimes(job.repeat)

  if (repeatText) {
    rows.push({ key: 'repeat', value: repeatText })
  }

  const skills = list(job.skills)

  if (skills) {
    rows.push({ key: 'skills', value: skills })
  }

  const toolsets = list(job.enabled_toolsets)

  if (toolsets) {
    rows.push({ key: 'enabledToolsets', value: toolsets })
  }

  const workdir = text(job.workdir)

  if (workdir) {
    rows.push({ key: 'workdir', value: workdir })
  }

  const contextFrom = list(job.context_from)

  if (contextFrom) {
    rows.push({ key: 'contextFrom', value: contextFrom })
  }

  const monitor = text(job.monitor_script) || text(job.monitor_url)

  if (monitor) {
    rows.push({ key: 'monitor', value: monitor })
  }

  if (job.attach_to_session === true) {
    rows.push({ key: 'attachToSession', value: 'on' })
  }

  const reasoning = text(job.reasoning_effort)

  if (reasoning) {
    rows.push({ key: 'reasoningEffort', value: reasoning })
  }

  const failureDeliver = text(job.failure_deliver)

  if (failureDeliver) {
    rows.push({ key: 'failureDeliver', value: failureDeliver })
  }

  return rows
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
export function cronEditorUpdates(
  values: CronEditorSaveValues,
  options: { initial?: CronJob | null; scriptOnlyJob: boolean }
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

  // Script-only jobs never run an agent, so the scheduler ignores model
  // overrides — leave whatever is stored untouched. For agent jobs, always
  // write both axes so resetting to "default" clears a previous pin (the
  // backend normalizes null/'' to "no override").
  if (!options.scriptOnlyJob) {
    updates.model = values.model.trim() || null
    updates.provider = values.provider.trim() || null
  }

  Object.assign(updates, cronAdvancedUpdates(values.advanced ?? emptyCronAdvancedValues(), options.initial ?? null))

  return updates
}

/** Advanced-settings payload subset: the ten fields the collapsed section
 *  owns. Both the create payload and the update payload accept these shapes
 *  (backend normalizes null/'' to unset). */
export interface CronAdvancedUpdateFields {
  attach_to_session?: boolean | null
  context_from?: string[]
  enabled_toolsets?: string[]
  failure_deliver?: null | string
  monitor_script?: null | string
  monitor_url?: null | string
  reasoning_effort?: null | string
  repeat?: null | number
  skills?: string[]
  workdir?: null | string
}

/** Build the advanced-settings update payload. Change-aware: fields the user
 *  did not touch are omitted so editing an old job never clobbers stored
 *  values the form never displayed (issue point 4). With no initial job
 *  (create mode) every non-empty field is sent. */
export function cronAdvancedUpdates(
  advanced: CronAdvancedValues,
  initial: CronJob | null
): CronAdvancedUpdateFields {
  const updates: CronAdvancedUpdateFields = {}
  const seed = initial ? seedCronAdvancedValues(initial) : emptyCronAdvancedValues()
  const changed = (key: keyof CronAdvancedValues): boolean => advanced[key] !== seed[key]

  const skills = parseCommaList(advanced.skills)

  if (!initial ? skills.length > 0 : changed('skills')) {
    updates.skills = skills
  }

  const toolsets = parseCommaList(advanced.enabledToolsets)

  if (!initial ? toolsets.length > 0 : changed('enabledToolsets')) {
    updates.enabled_toolsets = toolsets
  }

  const contextFrom = parseCommaList(advanced.contextFrom)

  if (!initial ? contextFrom.length > 0 : changed('contextFrom')) {
    updates.context_from = contextFrom
  }

  const workdir = advanced.workdir.trim()

  if (!initial ? Boolean(workdir) : changed('workdir')) {
    updates.workdir = workdir || null
  }

  const monitorScript = advanced.monitorScript.trim()
  const monitorUrl = advanced.monitorUrl.trim()

  if (!initial ? Boolean(monitorScript) : changed('monitorScript')) {
    updates.monitor_script = monitorScript || null
  }

  if (!initial ? Boolean(monitorUrl) : changed('monitorUrl')) {
    updates.monitor_url = monitorUrl || null
  }

  const repeat = parseRepeatTimes(advanced.repeat)

  if (!Number.isNaN(repeat) && (!initial ? repeat !== null : changed('repeat'))) {
    // null clears a stored repeat back to forever; the backend preserves the
    // completed counter on bare values.
    updates.repeat = repeat
  }

  if (!initial ? advanced.attachToSession : changed('attachToSession')) {
    updates.attach_to_session = advanced.attachToSession
  }

  const reasoning = advanced.reasoningEffort.trim().toLowerCase()

  if (!initial ? Boolean(reasoning) : changed('reasoningEffort')) {
    updates.reasoning_effort = reasoning || null
  }

  const failureDeliver = advanced.failureDeliver.trim()

  if (!initial ? Boolean(failureDeliver) : changed('failureDeliver')) {
    updates.failure_deliver = failureDeliver || null
  }

  return updates
}
