import { atom } from 'nanostores'

import { $gateway } from './gateway'
import { $activeSessionId } from './session'
import {
  runSessionControlAction,
  type SessionControlAction,
  type SessionControlDispatch
} from './session-control'

export type AutomationType = 'goal' | 'loop' | 'heartbeat'

export interface AutomationComposerState {
  open: boolean
  sessionId: string | null
  type: AutomationType
  submitting: boolean
  error: string | null
}

export interface GoalCreateArgs {
  prompt: string
  criteria?: string[]
  max_turns?: number
}

export interface LoopCreateArgs {
  prompt: string
  interval_seconds: number
  run_limit?: number
  stop_condition?: string
}

export interface HeartbeatCreateArgs {
  prompt: string
  interval_seconds: number
}

export type AutomationCreateArgs = GoalCreateArgs | LoopCreateArgs | HeartbeatCreateArgs

const EMPTY_STATE: AutomationComposerState = {
  open: false,
  sessionId: null,
  type: 'goal',
  submitting: false,
  error: null
}

export const $automationComposer = atom<AutomationComposerState>({ ...EMPTY_STATE })

$gateway.listen(() => $automationComposer.set({ ...EMPTY_STATE }))

function actionFor(type: AutomationType): SessionControlAction {
  switch (type) {
    case 'goal':
      return 'goal.create'

    case 'loop':
      return 'loop.create'

    case 'heartbeat':
      return 'heartbeat.create'
  }
}

/** Opens the composer dialog bound to the composer that was clicked — the
 *  session is captured here, never read from the global bot selection. */
export function openAutomationComposer(type: AutomationType = 'goal', sessionId?: string | null): void {
  const current = $automationComposer.get()

  if (current.submitting) {
    return
  }

  const sid = sessionId === undefined ? $activeSessionId.get() ?? null : sessionId

  $automationComposer.set({
    open: true,
    sessionId: sid,
    type,
    submitting: false,
    error: null
  })
}

export function closeAutomationComposer(): void {
  const current = $automationComposer.get()

  if (current.submitting) {
    return
  }

  $automationComposer.set({ ...EMPTY_STATE })
}

export function setAutomationComposerType(type: AutomationType): void {
  const current = $automationComposer.get()

  if (current.submitting) {
    return
  }

  $automationComposer.set({ ...current, type, error: null })
}

/** Closes the dialog when the active session no longer matches the captured
 *  one, so a stale completion can never land in a chat the user moved on to. */
export function invalidateOnSessionSwitch(): void {
  const current = $automationComposer.get()

  if (!current.open || !current.sessionId) {
    return
  }

  if ($activeSessionId.get() !== current.sessionId) {
    $automationComposer.set({ ...EMPTY_STATE })
  }
}

function boundedError(error: unknown): string {
  const message =
    error instanceof Error
      ? error.message
      : typeof error === 'object' && error !== null && 'message' in error
        ? String((error as { message: unknown }).message)
        : 'Failed to create automation'

  return message.trim().slice(0, 240) || 'Failed to create automation'
}

/** Runs a validated `session.control` creation action for the captured
 *  session. On success the dialog is closed and the resulting dispatch (which
 *  may be a `send` carrying the kickoff message for goals) is returned; on any
 *  failure the dialog stays open with the error surfaced. */
export async function submitAutomation(
  type: AutomationType,
  args: AutomationCreateArgs,
  onDispatch?: (dispatch: SessionControlDispatch) => Promise<void>
): Promise<SessionControlDispatch> {
  const current = $automationComposer.get()
  const sessionId = current.sessionId

  if (!sessionId) {
    const message = 'No active session'
    $automationComposer.set({ ...current, submitting: false, error: message })

    throw new Error(message)
  }

  if (!current.open || current.submitting || current.type !== type) {
    throw new Error("Automation request is no longer available")
  }

  const pending = { ...current, submitting: true, error: null }
  $automationComposer.set(pending)

  try {
    const dispatch = await runSessionControlAction(sessionId, actionFor(type), args)

    if ($automationComposer.get() !== pending) {
      throw new Error("Automation request belongs to a previous conversation")
    }

    await onDispatch?.(dispatch)

    if ($automationComposer.get() !== pending) {
      throw new Error("Automation request belongs to a previous conversation")
    }

    $automationComposer.set({ ...EMPTY_STATE })

    return dispatch
  } catch (error) {
    if ($automationComposer.get() === pending) {
      $automationComposer.set({ ...pending, submitting: false, error: boundedError(error) })
    }

    throw error
  }
}
