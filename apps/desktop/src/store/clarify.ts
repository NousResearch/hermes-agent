import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'
import { $activeSessionId } from './session'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

/**
 * Validate and normalize a choices array.
 *
 * Keeps non-blank, newline-free strings of length ≤ 200; drops everything else
 * and returns an empty array when nothing usable survives — the caller then
 * falls back to a free-text answer instead of dead buttons.
 */
export function normalizeChoices(choices: unknown): string[] {
  if (!Array.isArray(choices)) {
    return []
  }

  return choices.filter(
    (c): c is string => typeof c === 'string' && c.trim().length > 0 && c.length <= 200 && !c.includes('\n')
  )
}

/**
 * Structured warning for a clarify payload that arrived with choices but had
 * them all normalized away — keeps the remaining #69122 "no selectable choices"
 * triggers diagnosable in the field without dead constant fields.
 */
export function warnDroppedChoices(source: 'gateway' | 'tool_args', question: string, rawChoices: unknown): void {
  console.warn('[clarify] choices dropped after normalization', {
    choices_count: Array.isArray(rawChoices) ? rawChoices.length : 0,
    question_length: question.length,
    source
  })
}

export interface ClarifyInputState {
  draft: string
  focusLocked: boolean
  scrollTop: number
  selectedChoice: string | null
  selectionEnd: number | null
  selectionStart: number | null
}

export interface ClarifyTextareaPosition {
  scrollTop: number
  selectionEnd: number
  selectionStart: number
}

// Pending clarify requests keyed by the runtime session id that raised them.
// Storing per-session (instead of one shared slot) lets a *background* session
// park its clarify request while the user is looking at a different chat, then
// resolve it once they switch over — without a second concurrent clarify
// clobbering the first. A request with no session id lands under the empty key.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $clarifyRequests = atom<Record<string, ClarifyRequest>>({})

// The clarify request for the currently-viewed session. The inline ClarifyTool
// only ever mounts inside the active session's transcript, so it reads this
// focus-scoped view rather than reaching into the whole map.
export const $clarifyRequest = computed(
  [$clarifyRequests, $activeSessionId],
  (requests, activeId) => requests[keyFor(activeId)] ?? null
)

/** The clarify request for one specific session — the tile counterpart of the
 *  active-session `$clarifyRequest` view (same map, fixed key). */
export const sessionClarifyRequest = (sessionId: string | null) =>
  computed($clarifyRequests, requests => requests[keyFor(sessionId)] ?? null)

// Inline clarify state is kept outside the tool component because assistant
// stream updates can remount the tool while the user is typing. Every key is
// session-scoped so two chats asking the same question cannot share a draft.
export const $clarifyInputs = atom<Record<string, ClarifyInputState>>({})

function normalizeClarifyInput(input?: Partial<ClarifyInputState>): ClarifyInputState {
  return {
    draft: input?.draft ?? '',
    focusLocked: input?.focusLocked ?? false,
    scrollTop: input?.scrollTop ?? 0,
    selectedChoice: input?.selectedChoice ?? null,
    selectionEnd: input?.selectionEnd ?? null,
    selectionStart: input?.selectionStart ?? null
  }
}

function updateClarifyInput(key: string, patch: Partial<ClarifyInputState>): void {
  const current = $clarifyInputs.get()
  const previous = normalizeClarifyInput(current[key])
  const next = { ...previous, ...patch }

  if (
    previous.draft === next.draft &&
    previous.focusLocked === next.focusLocked &&
    previous.scrollTop === next.scrollTop &&
    previous.selectedChoice === next.selectedChoice &&
    previous.selectionEnd === next.selectionEnd &&
    previous.selectionStart === next.selectionStart
  ) {
    return
  }

  $clarifyInputs.set({ ...current, [key]: next })
}

export function clarifyInputKey(
  sessionId: string | null | undefined,
  requestId?: null | string,
  question?: string
): string {
  const sessionKey = sessionId?.trim() ?? ''
  const id = requestId?.trim()

  if (id) {
    return `session:${sessionKey}:request:${id}`
  }

  const normalizedQuestion = question?.trim()

  return normalizedQuestion ? `session:${sessionKey}:question:${normalizedQuestion}` : `session:${sessionKey}:pending`
}

function migrateClarifyInput(request: ClarifyRequest, previousRequest?: ClarifyRequest): void {
  const idKey = clarifyInputKey(request.sessionId, request.requestId, request.question)
  const questionKey = clarifyInputKey(request.sessionId, null, request.question)

  const previousKey =
    previousRequest?.question === request.question
      ? clarifyInputKey(request.sessionId, previousRequest.requestId, previousRequest.question)
      : null

  const current = $clarifyInputs.get()
  const sourceKeys = [...new Set([idKey, previousKey, questionKey].filter((key): key is string => Boolean(key)))]
  const persisted = sourceKeys.map(key => current[key]).find(Boolean)

  if (!persisted) {
    return
  }

  const next = { ...current }

  for (const key of sourceKeys) {
    delete next[key]
  }

  next[idKey] = {
    ...persisted,
    selectedChoice:
      persisted.selectedChoice && request.choices?.includes(persisted.selectedChoice) ? persisted.selectedChoice : null
  }

  $clarifyInputs.set(next)
}

export function setClarifyRequest(request: ClarifyRequest): void {
  const requests = $clarifyRequests.get()
  const requestKey = keyFor(request.sessionId)

  migrateClarifyInput(request, requests[requestKey])
  $clarifyRequests.set({ ...requests, [requestKey]: request })
}

export function clearClarifyRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $clarifyRequests.get()

  // Targeted clear when the caller knows the session (the common path from the
  // inline ClarifyTool answering its own request).
  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return
    }

    clearClarifyInput(clarifyInputKey(current.sessionId, current.requestId, current.question))
    clearClarifyInput(clarifyInputKey(current.sessionId, null, current.question))

    const next = { ...requests }
    delete next[key]
    $clarifyRequests.set(next)

    return
  }

  // Fallback with no session hint: drop every entry matching the request id
  // (or clear all when none is given).
  const next: Record<string, ClarifyRequest> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      changed = true
      clearClarifyInput(clarifyInputKey(value.sessionId, value.requestId, value.question))
      clearClarifyInput(clarifyInputKey(value.sessionId, null, value.question))
    }
  }

  if (changed) {
    $clarifyRequests.set(next)
  }
}

/** Whether `sessionId` has a clarify parked on it right now (imperative read —
 *  the composer checks this on Enter, not on every render). */
export const hasClarifyRequest = (sessionId: string | null | undefined): boolean =>
  Boolean($clarifyRequests.get()[keyFor(sessionId)])

/**
 * Answer `sessionId`'s pending clarify with an empty answer (a skip) and drop it
 * locally, resolving to whether there was one to skip.
 *
 * The composer uses this when the user types a real message instead of picking
 * an option: a clarify blocks the agent inside its tool batch, so leaving it
 * unanswered would park the follow-up until the server-side clarify timeout
 * (default 5 min) — the message looks sent and nothing happens. Skipping lets
 * the tool return and the turn carry on with the user's actual words.
 *
 * An empty answer is the same thing the card's own Skip button sends, and
 * `clarify.respond` is `allow_expired`, so racing the timeout is harmless.
 */
export async function skipClarifyRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  if (!request) {
    return false
  }

  // Clear first: the answer is already decided, and an in-flight RPC must not
  // leave a live card the user can answer a second time.
  clearClarifyRequest(request.requestId, request.sessionId)

  try {
    await $gateway.get()?.request('clarify.respond', { request_id: request.requestId, answer: '' })
  } catch {
    // The tool times out on its own; a failed skip must never swallow the
    // message the user is actually sending.
  }

  return true
}

export function clearClarifyInput(key: string): void {
  const current = $clarifyInputs.get()

  if (!current[key]) {
    return
  }

  const { [key]: _cleared, ...rest } = current

  $clarifyInputs.set(rest)
}

export function setClarifyDraft(key: string, draft: string, position?: ClarifyTextareaPosition): void {
  updateClarifyInput(key, {
    draft,
    ...position,
    ...(draft.trim() ? { selectedChoice: null } : {})
  })
}

export function setClarifySelectedChoice(key: string, selectedChoice: string | null): void {
  updateClarifyInput(key, {
    selectedChoice,
    ...(selectedChoice ? { draft: '', focusLocked: false } : {})
  })
}

export function setClarifyFocusLocked(key: string, focusLocked: boolean): void {
  updateClarifyInput(key, { focusLocked })
}

export function setClarifyTextareaPosition(key: string, position: ClarifyTextareaPosition): void {
  updateClarifyInput(key, position)
}
