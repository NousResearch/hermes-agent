/**
 * Open `setup_beat` requests: the onboarding chat's card for one beat, one per session.
 *
 * The setup_beat tool row reads this to draw the beat's card. A pick answers the request with
 * `{answer}`. Composer text answers only the name card, with `{said}` (the words are the name); a
 * message sent while any other card waits closes it with `{wrote_instead}`, the backend ends that
 * turn, and the composer sends the message as the next turn. The backend's `request.cancel`
 * (timeout, Stop, Skip setup) clears the entry by request id, so a late cancel never erases a newer
 * card the same session raised; the setup_beat call's
 * completion clears it by tool call id, which covers a card answered in another window.
 */

import type { ToolCallMessagePart } from '@assistant-ui/react'
import type {
  SetupBeat,
  SetupBeatResult,
  SetupBuildAnswer,
  SetupConnectorsAnswer,
  SetupLayoutAnswer,
  SetupNameAnswer,
  SetupPluginsAnswer,
  SetupThemeAnswer,
  SetupTourAnswer
} from '@hermes/shared'
import { atom } from 'nanostores'

import { recordOf } from '@/lib/connector-tools'

import { registerParkedCardAnswer } from './parked-card-answers'
import { respondToServerRequest } from './server-requests'

export interface BeatOption {
  id: string
  label: string
}

/** A beat's card data (`agent/onboarding_flow.py::BEATS`, `tools/setup_beat_tool.py::_card`): name `{suggested?}`
 *  (the account's full name), layout/tour/build `{options}`, connectors `{suggested}` (slugs), plugins `{plugins}`. */
export interface SetupBeatCard {
  options?: BeatOption[]
  plugins?: string[]
  suggested?: string | string[]
}

/** The picks a card answers with: one of the shared per-beat answers (`SetupBeatResult.answer`). */
export type SetupBeatAnswer = NonNullable<SetupBeatResult['answer']>

/** An answer the receipt reads by field, whichever beat it came from. Built from the shared answers, so a
 *  field the contract adds or renames reaches the receipt. */
export type SetupBeatAnswerFields = Partial<
  SetupBuildAnswer &
    SetupConnectorsAnswer &
    SetupLayoutAnswer &
    SetupNameAnswer &
    SetupPluginsAnswer &
    SetupThemeAnswer &
    SetupTourAnswer
>

/** What `tools/setup_beat_tool.py` returns to the model, as the transcript's tool row carries it. */
export interface SetupBeatToolResult {
  answer?: SetupBeatAnswerFields
  beat?: SetupBeat
  completed?: boolean
  error?: string
  said?: string
  skipped_all?: boolean
  timed_out?: boolean
  wrote_instead?: boolean
}

/** A setup_beat tool row's result: the object a live row carries, or the JSON text of a reloaded one. */
export function setupBeatResult(result: ToolCallMessagePart['result']): SetupBeatToolResult {
  // SAFETY: tools/setup_beat_tool.py::_result builds every result from an agent/onboarding_flow.py step.
  return recordOf(result) as SetupBeatToolResult
}

export interface SetupBeatRequest {
  beat: SetupBeat
  card: SetupBeatCard
  question: string
  requestId: string
  sessionId: string
  /** The setup_beat call the card belongs to; only that transcript row draws it. */
  toolCallId: null | string
}

export const $setupBeatRequests = atom<Record<string, SetupBeatRequest>>({})

export function parkSetupBeat(request: SetupBeatRequest): void {
  $setupBeatRequests.set({ ...$setupBeatRequests.get(), [request.sessionId]: request })
}

/** Drop the card for `requestId`; a no-op when the session's card is a newer one. */
export function clearSetupBeat(requestId: string, sessionId: string): boolean {
  const { [sessionId]: current, ...others } = $setupBeatRequests.get()

  if (current?.requestId !== requestId) {
    return false
  }

  $setupBeatRequests.set(others)

  return true
}

/** The run ended: the session's card, if one still waits, is done. */
export function dropSetupBeat(sessionId: string): void {
  const request = $setupBeatRequests.get()[sessionId]

  if (request) {
    clearSetupBeat(request.requestId, sessionId)
  }
}

// The card clears only once the answer is sent, so a failed send leaves it on screen to answer again.
function respond(
  sessionId: string,
  result: { answer: SetupBeatAnswer } | Record<string, never> | { said: string } | { wrote_instead: true }
): boolean {
  const request = $setupBeatRequests.get()[sessionId]

  if (!request || !respondToServerRequest(request.requestId, { ...result })) {
    return false
  }

  clearSetupBeat(request.requestId, sessionId)

  return true
}

/** Answer the session's card with the beat's picks. False when no card waits. */
export function answerSetupBeat(sessionId: string, answer: SetupBeatAnswer): boolean {
  return respond(sessionId, { answer })
}

/** Composer text typed while the name card waits is the name. No other card takes typed words. */
export function sayToSetupBeat(sessionId: string, text: string): null | Promise<boolean> {
  return text.trim() && $setupBeatRequests.get()[sessionId]?.beat === 'name'
    ? Promise.resolve(respond(sessionId, { said: text.trim() }))
    : null
}

/** A message sent instead of an answer closes the card; the backend ends its turn on that result. */
export function yieldSetupBeatToMessage(sessionId: string): boolean {
  return respond(sessionId, { wrote_instead: true })
}

registerParkedCardAnswer('setup_beat', {
  onboardingOnly: true,
  $requests: $setupBeatRequests,
  answer: sayToSetupBeat,
  // An empty answer closes the card unanswered, so a slash command can run.
  skip: sessionId => respond(sessionId, {}),
  waiting: sessionId => Boolean($setupBeatRequests.get()[sessionId]),
  yieldToMessage: yieldSetupBeatToMessage
})
