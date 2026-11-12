import type { CronJob, CronJobCreatePayload, CronJobUpdates } from '@/hermes'

import {
  cronEditorUpdates,
  cronModelChoiceValue,
  jobIsScriptOnly,
  parseCronModelChoiceValue
} from '../cron/cron-job-model'
import { type CatchUpChoice, catchUpChoice, catchUpValue } from '../cron/run-control'

import { remembersLastRun, runLimit, withRememberLastRun } from './routine-model'
import { composeSchedule, defaultPick, pickForSchedule, type SchedulePick } from './routine-schedule'

// The one editor's form state (R4) and its translation to the existing create/update payloads. Edits
// go through the same `cronEditorUpdates` the classic editor uses, so a field the user did not touch
// is never rewritten; the redesign's extra fields (run limit, remember last run) are sent only when
// they changed.

/** Radix selects reject '', so "follow the default model" rides this sentinel (as in the classic editor). */
export const MODEL_DEFAULT = '__default__'

export interface RoutineDraft {
  profile: string
  name: string
  prompt: string
  pick: SchedulePick
  deliver: string
  /** `cronModelChoiceValue(provider, model)` or MODEL_DEFAULT. */
  model: string
  catchUp: CatchUpChoice
  /** Digits, '' = no limit. */
  runLimit: string
  remember: boolean
}

export function blankDraft(profile: string, nowMs = Date.now()): RoutineDraft {
  return {
    catchUp: 'default',
    deliver: 'local',
    model: MODEL_DEFAULT,
    name: '',
    pick: defaultPick(nowMs),
    profile,
    prompt: '',
    remember: false,
    runLimit: ''
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
    runLimit: limit === null ? '' : String(limit)
  }
}

export type DraftError = 'prompt' | 'schedule'

export function validateDraft(draft: RoutineDraft, job?: CronJob): DraftError | null {
  if (!composeSchedule(draft.pick)) {
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
    schedule: composeSchedule(draft.pick),
    ...(draft.name.trim() ? { name: draft.name.trim() } : {}),
    ...(model ? { model, provider: provider || undefined } : {}),
    ...(catchUp === null ? {} : { catch_up: catchUp }),
    ...(limit === null ? {} : { repeat: limit }),
    ...(draft.remember ? { context_from: ['self'] } : {})
  }
}

export function updatePayload(draft: RoutineDraft, job: CronJob): CronJobUpdates {
  const before = draftFromJob(job)
  const schedule = composeSchedule(draft.pick)

  const updates = cronEditorUpdates(
    {
      ...modelFields(draft),
      catchUp: draft.catchUp === before.catchUp ? undefined : catchUpValue(draft.catchUp),
      deliver: draft.deliver || 'local',
      name: draft.name.trim(),
      prompt: draft.prompt,
      // An untouched schedule is sent back as stored, so a natural phrase is never re-parsed.
      schedule: schedule === composeSchedule(before.pick) ? (job.schedule?.expr ?? schedule) : schedule
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
