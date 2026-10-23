import { useCallback, useState } from 'react'

import { queueKickoffIfSessionBusy } from '@/app/session/hooks/use-prompt-actions/queue-if-busy'
import type { Translations } from '@/i18n/types'
import type { AutomationCreateArgs, AutomationType } from '@/store/automation-composer'
import type { SessionControlDispatch, SessionControlEntry } from '@/store/session-control'

export type AutomationCopy = Translations['automationComposer']

export type OnSubmitText = (
  value: string,
  options?: { displayKind?: 'hidden'; sessionId?: string | null }
) => Promise<boolean> | boolean

/** Everything the dialog's per-type field groups read or write, in one object so a
 *  field group takes two props instead of thirteen. */
export interface AutomationFormValues {
  criteria: string[]
  draftCriterion: string
  interval: string
  maxTurns: string
  prompt: string
  runLimit: string
  stopCondition: string
}

export type AutomationFormField = keyof AutomationFormValues

export interface AutomationFormApi {
  values: AutomationFormValues
  set: (field: AutomationFormField, value: string) => void
  setCriteria: (criteria: string[]) => void
  addCriterion: () => void
  removeCriterion: (index: number) => void
  replace: (values: Partial<AutomationFormValues>) => void
}

const EMPTY_FORM: AutomationFormValues = {
  criteria: [],
  draftCriterion: '',
  interval: '',
  maxTurns: '',
  prompt: '',
  runLimit: '',
  stopCondition: ''
}

export function useAutomationForm(): AutomationFormApi {
  const [values, setValues] = useState(EMPTY_FORM)

  const set = useCallback((field: AutomationFormField, value: string) => {
    setValues(current => ({ ...current, [field]: value }))
  }, [])

  const setCriteria = useCallback((criteria: string[]) => {
    setValues(current => ({ ...current, criteria }))
  }, [])

  const addCriterion = useCallback(() => {
    setValues(current => {
      const value = current.draftCriterion.trim()

      return value ? { ...current, criteria: [...current.criteria, value], draftCriterion: '' } : current
    })
  }, [])

  const removeCriterion = useCallback((index: number) => {
    setValues(current => ({ ...current, criteria: current.criteria.filter((_, i) => i !== index) }))
  }, [])

  const replace = useCallback((patch: Partial<AutomationFormValues>) => {
    setValues(current => ({ ...current, ...patch }))
  }, [])

  return { values, set, setCriteria, addCriterion, removeCriterion, replace }
}

/** A rejected control action (a busy Goal) is retryable and stays visible as `actionError`;
 *  only a connection/read failure or an unsupported backend makes the controls unavailable. */
export function isUnavailable(entry: SessionControlEntry | undefined): boolean {
  return Boolean(entry) && (entry?.capability === 'unsupported' || Boolean(entry?.error))
}

/** The automation already on this session of the requested type, but never in edit mode —
 *  editing must not be blocked by the automation being edited. */
export function existingAutomation(
  entry: SessionControlEntry | undefined,
  type: AutomationType,
  isEdit: boolean
) {
  return isEdit ? undefined : entry?.snapshot?.[type]
}

/** Legacy backends omit `loop_min_interval_seconds`; a missing minimum never blocks a submit. */
export function loopIntervalTooShort(
  type: AutomationType,
  interval: string,
  min: number | undefined
): boolean {
  return type === 'loop' && interval !== '' && min !== undefined && Number(interval) < min
}

export function submitBlocked(input: {
  existing: unknown
  intervalTooShort: boolean
  isEdit: boolean
  pausing: boolean
  prompt: string
  sessionId: string | null
  submitting: boolean
  unavailable: boolean
}): boolean {
  const duplicate = !input.isEdit && Boolean(input.existing)

  return (
    input.submitting ||
    input.pausing ||
    !input.sessionId ||
    duplicate ||
    input.unavailable ||
    !input.prompt.trim() ||
    input.intervalTooShort
  )
}

export function pauseBlocked(input: {
  pausing: boolean
  sessionId: string | null
  submitting: boolean
  unavailable: boolean
}): boolean {
  return input.pausing || input.submitting || !input.sessionId || input.unavailable
}

const SUBMIT_KEYS = {
  goal: { create: 'startGoal', edit: 'saveGoal' },
  loop: { create: 'startLoop', edit: 'saveLoop' },
  heartbeat: { create: 'createHeartbeat', edit: 'saveHeartbeat' }
} as const satisfies Record<AutomationType, { create: keyof AutomationCopy; edit: keyof AutomationCopy }>

const FIRST_RUN_KEYS = {
  goal: 'firstRunGoal',
  loop: 'firstRunLoop',
  heartbeat: 'firstRunHeartbeat'
} as const satisfies Record<AutomationType, keyof AutomationCopy>

const PAUSE_KEYS = {
  goal: 'pauseGoal',
  loop: 'pauseLoop',
  heartbeat: 'pauseHeartbeat'
} as const satisfies Record<AutomationType, keyof AutomationCopy>

const IDLE_KEYS = { loop: 'idleLoop', heartbeat: 'idleHeartbeat' } as const

export interface AutomationLabels {
  firstRun: string
  idle: string
  pause: string
  submit: string
}

export function automationLabels(copy: AutomationCopy, type: AutomationType, isEdit: boolean): AutomationLabels {
  return {
    firstRun: copy[FIRST_RUN_KEYS[type]],
    idle: type === 'goal' ? '' : copy[IDLE_KEYS[type]],
    pause: copy[PAUSE_KEYS[type]],
    submit: copy[SUBMIT_KEYS[type][isEdit ? 'edit' : 'create']]
  }
}

export function buildArgs(
  type: AutomationType,
  isEdit: boolean,
  values: Pick<AutomationFormValues, 'criteria' | 'interval' | 'maxTurns' | 'prompt' | 'runLimit' | 'stopCondition'>
): AutomationCreateArgs {
  if (type === 'goal') {
    return {
      prompt: values.prompt,
      criteria: values.criteria,
      ...(values.maxTurns ? { max_turns: Number(values.maxTurns) } : {})
    }
  }

  if (type === 'loop') {
    const hasRunLimit = values.runLimit !== ''

    const runLimitValue = hasRunLimit ? Number(values.runLimit) : 0

    return {
      prompt: values.prompt,
      interval_seconds: Number(values.interval),
      ...(isEdit ? { run_limit: runLimitValue } : hasRunLimit ? { run_limit: runLimitValue } : {}),
      ...(values.stopCondition.trim() ? { stop_condition: values.stopCondition.trim() } : {})
    }
  }

  return {
    prompt: values.prompt,
    interval_seconds: Number(values.interval)
  }
}

/** A goal's create dispatch is a `send`: the backend persisted the goal and handed back the
 *  continuation prompt. Mirror the goal card's resume path — queue it while the session is
 *  busy, otherwise submit it and surface a kickoff failure the user can retry. */
export async function deliverKickoff(
  dispatch: SessionControlDispatch | null | undefined,
  options: { copy: AutomationCopy; onSubmitText: OnSubmitText; sessionId: string }
): Promise<void> {
  if (dispatch?.type !== 'send' || !dispatch.message) {
    return
  }

  const queued = queueKickoffIfSessionBusy({
    displayText: dispatch.display ?? undefined,
    sessionId: options.sessionId,
    text: dispatch.message
  })

  // 'queued' already parked the prompt on the composer queue; only a refusal to queue or a
  // failed direct submit is a kickoff failure the user must see.
  if (queued === 'busy') {
    throw new Error(options.copy.kickoffFailure)
  }

  if (queued === 'idle' && !(await options.onSubmitText(dispatch.message, { displayKind: 'hidden', sessionId: options.sessionId }))) {
    throw new Error(options.copy.kickoffFailure)
  }
}
