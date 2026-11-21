import type { AutomationBlueprint, CronJob, CronJobCreatePayload, CronJobUpdates } from '@/hermes'

import {
  cronEditorUpdates,
  cronModelChoiceValue,
  jobIsScriptOnly,
  parseCronModelChoiceValue
} from '../cron/cron-job-model'
import { type CatchUpChoice, catchUpChoice, catchUpValue } from '../cron/run-control'

import { remembersLastRun, runLimit, withRememberLastRun } from './routine-model'
import { composeSchedule, pickForSchedule, type SchedulePick } from './routine-schedule'

// The routine sheet's form state and its translation to the existing create/update payloads. Edits
// go through the same `cronEditorUpdates` the classic editor uses, so a field the user did not touch
// is never rewritten; the redesign's extra fields (run limit, remember last run) are sent only when
// they changed.

/** Radix selects reject '', so "follow the default model" rides this sentinel (as in the classic editor). */
export const MODEL_DEFAULT = '__default__'

export interface RoutineDraft {
  profile: string
  name: string
  prompt: string
  /** null until a frequency is picked: a new routine starts with none. */
  pick: null | SchedulePick
  deliver: string
  /** `cronModelChoiceValue(provider, model)` or MODEL_DEFAULT. */
  model: string
  catchUp: CatchUpChoice
  /** Digits, '' = no limit. */
  runLimit: string
  remember: boolean
  /** A template's own skills, which its instruction relies on. */
  skills: string[]
}

export function blankDraft(profile: string): RoutineDraft {
  return {
    catchUp: 'default',
    deliver: 'local',
    model: MODEL_DEFAULT,
    name: '',
    pick: null,
    profile,
    prompt: '',
    remember: false,
    runLimit: '',
    skills: []
  }
}

/** A template's routine, its defaults filled in: the job the backend says they make (`prefill`), or,
 *  from a backend that predates it, its name and description. "origin" (the chat a routine was set up
 *  from) means nothing to a sheet, so it starts with nothing selected: kept in Runs. */
export function draftFromTemplate(template: AutomationBlueprint, base: RoutineDraft, nowMs = Date.now()): RoutineDraft {
  const prefill = template.prefill
  const deliver = prefill?.deliver && prefill.deliver !== 'origin' ? prefill.deliver : 'local'

  return {
    ...base,
    deliver,
    name: prefill?.name || template.title,
    pick: prefill ? pickForSchedule({ expr: prefill.schedule, kind: 'cron' }, prefill.schedule, nowMs) : base.pick,
    prompt: prefill?.prompt || template.description,
    skills: prefill?.skills ?? []
  }
}

export function draftFromJob(job: CronJob, nowMs = Date.now()): RoutineDraft {
  const model = (job.model ?? '').trim()
  const limit = runLimit(job)

  return {
    catchUp: catchUpChoice(job.catch_up),
    deliver: (job.deliver ?? '').trim() || 'local',
    model: model ? cronModelChoiceValue((job.provider ?? '').trim(), model) : MODEL_DEFAULT,
    name: (job.name ?? '').trim(),
    pick: pickForSchedule(job.schedule, job.schedule_display ?? '', nowMs),
    profile: job.profile ?? '',
    prompt: job.prompt ?? '',
    remember: remembersLastRun(job),
    runLimit: limit === null ? '' : String(limit),
    skills: []
  }
}

export type DraftError = 'prompt' | 'schedule'

export function validateDraft(draft: RoutineDraft, job?: CronJob): DraftError | null {
  if (!draft.pick || !composeSchedule(draft.pick)) {
    return 'schedule'
  }

  return !draft.prompt.trim() && !(job && jobIsScriptOnly(job)) ? 'prompt' : null
}

function limitValue(text: string): null | number {
  const value = Number.parseInt(text, 10)

  return Number.isFinite(value) && value > 0 ? value : null
}

function modelFields(draft: RoutineDraft): { model: string; provider: string } {
  const choice = draft.model === MODEL_DEFAULT ? null : parseCronModelChoiceValue(draft.model)

  return { model: choice?.model ?? '', provider: choice?.provider ?? '' }
}

export function createPayload(draft: RoutineDraft): CronJobCreatePayload {
  const { model, provider } = modelFields(draft)
  const limit = limitValue(draft.runLimit)
  const catchUp = catchUpValue(draft.catchUp)

  return {
    deliver: draft.deliver || 'local',
    prompt: draft.prompt.trim(),
    schedule: draft.pick ? composeSchedule(draft.pick) : '',
    ...(draft.name.trim() ? { name: draft.name.trim() } : {}),
    ...(draft.skills.length ? { skills: draft.skills } : {}),
    ...(model ? { model, provider: provider || undefined } : {}),
    ...(catchUp === null ? {} : { catch_up: catchUp }),
    ...(limit === null ? {} : { repeat: limit }),
    ...(draft.remember ? { context_from: ['self'] } : {})
  }
}

export function updatePayload(draft: RoutineDraft, job: CronJob): CronJobUpdates {
  const before = draftFromJob(job)
  const schedule = draft.pick ? composeSchedule(draft.pick) : ''
  const stored = before.pick ? composeSchedule(before.pick) : ''

  const updates = cronEditorUpdates(
    {
      ...modelFields(draft),
      catchUp: draft.catchUp === before.catchUp ? undefined : catchUpValue(draft.catchUp),
      deliver: draft.deliver || 'local',
      name: draft.name.trim(),
      prompt: draft.prompt,
      // An untouched schedule is sent back as stored, so a natural phrase is never re-parsed.
      schedule: schedule === stored ? (job.schedule?.expr ?? schedule) : schedule
    },
    { scriptOnlyJob: jobIsScriptOnly(job) }
  )

  if (draft.runLimit !== before.runLimit) {
    updates.repeat = limitValue(draft.runLimit)
  }

  if (draft.remember !== before.remember) {
    updates.context_from = withRememberLastRun(job.context_from, draft.remember)
  }

  return updates
}
