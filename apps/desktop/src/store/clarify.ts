import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'
import { $activeSessionId } from './session'

export interface ClarifyQuestion {
  /** Server-generated wire id (q0..qN) — clarify.respond keys answers by it. */
  qid: string
  question: string
  choices: string[] | null
  multiSelect: boolean
}

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  multiSelect: boolean
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  choicesMalformed?: boolean
  sessionId: string | null
  /** Batch (multi-question) clarify: present instead of question/choices. */
  questions?: ClarifyQuestion[]
  /** Answers already locked server-side (reconnect replay): qid → answer. */
  lockedAnswers?: Record<string, string>
}

/**
 * The backend labels the agent's recommended option by appending this to the
 * first choice (`tools/clarify_tool.py::mark_recommended`). The renderer never
 * writes it — it only styles it, and discounts it when measuring a choice so a
 * long option isn't dropped for length the label added.
 */
export const RECOMMENDED_LABEL = '(Recommended)'

export const bareChoice = (choice: string): string =>
  choice.endsWith(RECOMMENDED_LABEL) ? choice.slice(0, -RECOMMENDED_LABEL.length).trim() : choice

/**
 * Validate and normalize a choices array.
 *
 * Keeps non-blank, newline-free strings of length ≤ 200; drops everything else
 * and returns an empty array when nothing usable survives. Callers can use
 * `hasMalformedStructuredChoices` to distinguish that failure from a
 * deliberately open-ended prompt.
 */
export function normalizeChoices(choices: unknown): string[] {
  if (!Array.isArray(choices)) {
    return []
  }

  return choices.filter(
    (c): c is string => typeof c === 'string' && c.trim().length > 0 && bareChoice(c).length <= 200 && !c.includes('\n')
  )
}

/** Mirror the backend's permissive `_flatten_choice` just far enough to decide
 * whether a raw tool argument can become a usable label before the gateway
 * emits its normalized strings. The renderer still displays only strings from
 * `normalizeChoices`; this helper prevents supported dict/scalar arguments from
 * being mistaken for an all-invalid choice set during the tool.start race. */
function flattenStructuredChoice(choice: unknown): string {
  if (choice == null) {
    return ''
  }

  if (typeof choice === 'string') {
    return choice.trim()
  }

  if (Array.isArray(choice)) {
    return choice.map(flattenStructuredChoice).join(' ').trim()
  }

  if (typeof choice === 'object') {
    const row = choice as Record<string, unknown>

    for (const key of ['label', 'description', 'text', 'title']) {
      const value = row[key]

      if (typeof value === 'string' && value.trim()) {
        return value.trim()
      }
    }

    return ''
  }

  return String(choice).trim()
}

/** True when a caller attempted a structured prompt but every option was
 * rejected. Omitted, null, and explicitly empty choices remain open-ended. */
export function hasMalformedStructuredChoices(
  rawChoices: unknown,
  normalizedChoices = normalizeChoices(rawChoices)
): boolean {
  return (
    Array.isArray(rawChoices) &&
    rawChoices.length > 0 &&
    normalizedChoices.length === 0 &&
    normalizeChoices(rawChoices.map(flattenStructuredChoice)).length === 0
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

/**
 * Validate and normalize a batch clarify payload's `questions` array.
 *
 * Keeps entries with a non-blank string `qid` and `question`; per-question
 * choices go through `normalizeChoices` (all-blank → open-ended) and
 * multi_select is only honored alongside surviving choices. Returns an empty
 * array when nothing usable remains — the caller treats that as "not a
 * batch" instead of rendering an unanswerable form.
 */
export function normalizeQuestions(questions: unknown): ClarifyQuestion[] {
  if (!Array.isArray(questions)) {
    return []
  }

  const normalized: ClarifyQuestion[] = []

  for (const entry of questions) {
    if (typeof entry !== 'object' || entry === null) {
      continue
    }

    const row = entry as Record<string, unknown>
    const qid = typeof row.qid === 'string' ? row.qid.trim() : ''
    const question = typeof row.question === 'string' ? row.question.trim() : ''

    if (!qid || !question) {
      continue
    }

    const choices = normalizeChoices(row.choices)

    normalized.push({
      choices: choices.length > 0 ? choices : null,
      multiSelect: row.multi_select === true && choices.length > 0,
      qid,
      question
    })
  }

  return normalized
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

export function setClarifyRequest(request: ClarifyRequest): void {
  $clarifyRequests.set({ ...$clarifyRequests.get(), [keyFor(request.sessionId)]: request })
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
