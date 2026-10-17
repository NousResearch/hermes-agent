import { atom, computed } from 'nanostores'

import { translateNow } from '@/i18n'

import { $gateway } from './gateway'
import { notifyError } from './notifications'
import { registerParkedCardAnswer } from './parked-card-answers'
import { forgetServerRequest, hasOpenServerRequest, respondToServerRequest } from './server-requests'
import { $activeSessionId } from './session'
import { requestForOwnedSession } from './session-states'

export interface ClarifyQuestion {
  qid: string
  question: string
  choices: string[] | null
  multiSelect: boolean
}

export interface ClarifyRequest {
  requestId: string
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
  questions: ClarifyQuestion[]
  /** Answers already locked server-side (reconnect replay): qid → answer, null = skipped. */
  lockedAnswers?: Record<string, null | string>
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
 * Per-choice display cap. The clarify tool enforces the same limit at the
 * source (`tools/clarify_tool.py::MAX_CHOICE_CHARS`) and declares it in the
 * schema, so an over-limit choice is rejected before any surface renders;
 * this filter is the last line of defence against a stale/other producer.
 * Not a one-line label limit — long option text wraps (`wrap-anywhere`),
 * newlines are kept so option reasons can read as multiple lines.
 */
export const MAX_CHOICE_CHARS = 8000

/**
 * Validate and normalize a choices array.
 *
 * Keeps non-blank strings (newlines allowed) whose bare text is within
 * MAX_CHOICE_CHARS; drops everything else and returns an empty array when
 * nothing usable survives — the caller then falls back to a free-text
 * answer instead of dead buttons.
 */
export function normalizeChoices(choices: unknown): string[] {
  if (!Array.isArray(choices)) {
    return []
  }

  return choices.filter(
    (c): c is string => typeof c === 'string' && c.trim().length > 0 && bareChoice(c).length <= MAX_CHOICE_CHARS
  )
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
  forgetSettledClarify(request.requestId)
}

/**
 * Tool results the renderer already knows for requests it answered, keyed by
 * request id. The card settles from this the moment it is answered, so a skip
 * or a typed answer never falls back to a bare tool row when the tool's own
 * `tool.complete` is late or never lands (a stopped turn).
 */
export const $settledClarifyResults = atom<Record<string, Record<string, unknown>>>({})

function forgetSettledClarify(requestId: string): void {
  const settled = $settledClarifyResults.get()

  if (requestId in settled) {
    const next = { ...settled }
    delete next[requestId]
    $settledClarifyResults.set(next)
  }
}

function settleClarify(request: ClarifyRequest, result: Record<string, unknown>): void {
  $settledClarifyResults.set({ ...$settledClarifyResults.get(), [request.requestId]: result })
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

/** Clear a stale card at a turn boundary, but keep it while its backend request is still waiting. */
export function clearSettledClarifyRequest(sessionId: string | null): void {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  if (request && !hasOpenServerRequest(request.requestId)) {
    clearClarifyRequest(request.requestId, sessionId)
  }
}

/** A locked answer as the server's result carries it: a multi-select answer is stored as a JSON list. */
function lockedAnswer(question: ClarifyQuestion, raw: string): string | string[] {
  if (!question.multiSelect) {
    return raw
  }

  try {
    const parsed: unknown = JSON.parse(raw)

    return Array.isArray(parsed) ? parsed.map(String) : raw
  } catch {
    return raw
  }
}

/**
 * Skip a parked card: clear it, answer its request with no pick, and settle it
 * as skipped. The card's Skip button uses this, and so does the composer for a
 * message that cannot be an answer (a slash command, attachments).
 */
export function skipClarify(request: ClarifyRequest): void {
  // Clear first: the answer is already decided, and an in-flight RPC must not
  // leave a live card the user can answer a second time.
  clearClarifyRequest(request.requestId, request.sessionId)

  respondToServerRequest(request.requestId, {})

  settleClarify(request, {
    outcome: 'cancelled',
    // Answers locked server-side before a reconnect stand: the server merges them into its result too.
    responses: request.questions.map(question => {
      const locked = request.lockedAnswers?.[question.qid]

      return locked
        ? { question: question.question, status: 'answered', user_response: lockedAnswer(question, locked) }
        : { question: question.question, status: 'unanswered', user_response: null }
    })
  })
}

export async function skipClarifyRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  if (!request) {
    return false
  }

  skipClarify(request)

  return true
}

/** Answer the clarify parked on `sessionId` with composer text: the answer to its first question not yet locked. */
function answerClarifyText(sessionId: string, text: string): null | Promise<boolean> {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  if (!request) {
    return null
  }

  return lockTypedAnswer(request, text)
}

registerParkedCardAnswer('clarify', {
  $requests: $clarifyRequests,
  answer: answerClarifyText,
  waiting: hasClarifyRequest
})

/** The choice typed text names (any case, as the messaging path matches labels), else the words themselves. */
function typedAnswer(question: ClarifyQuestion, text: string): string {
  const typed = text.toLowerCase()
  const answer = question.choices?.map(bareChoice).find(choice => choice.toLowerCase() === typed) ?? text

  return question.multiSelect ? JSON.stringify([answer]) : answer
}

/**
 * A request's typed answers are locked one at a time in answer order, as the card's Confirm does: an
 * answer is not sent after an earlier one failed, so the server never holds a later answer without it.
 */
const lockChains = new Map<string, Promise<boolean>>()

function lockTypedAnswer(request: ClarifyRequest, text: string): null | Promise<boolean> {
  const gateway = $gateway.get()
  const question = request.questions.find(entry => request.lockedAnswers?.[entry.qid] === undefined)

  if (!gateway || !question) {
    return null
  }

  const answer = typedAnswer(question, text)
  const lockedAnswers = { ...request.lockedAnswers, [question.qid]: answer }
  const last = request.questions.every(entry => lockedAnswers[entry.qid] !== undefined)

  // The store changes before the RPC returns, so a second Enter locks the next question, not this one again.
  if (last) {
    clearClarifyRequest(request.requestId, request.sessionId)
    settleClarify(request, {
      outcome: 'submitted',
      responses: request.questions.map(entry => {
        const locked = lockedAnswers[entry.qid]

        return locked
          ? { question: entry.question, status: 'answered', user_response: lockedAnswer(entry, locked) }
          : { question: entry.question, status: 'skipped', user_response: null }
      })
    })
  } else {
    setClarifyRequest({ ...request, lockedAnswers })
  }

  const previous = lockChains.get(request.requestId) ?? Promise.resolve(true)

  const sent = previous.then(async earlierSaved => {
    if (earlierSaved) {
      try {
        const reply = await requestForOwnedSession<{ status?: string }>(
          request.sessionId,
          <R>(...args: Parameters<typeof gateway.request>) => gateway.request<R>(...args),
          'clarify.lock',
          { answer, question_id: question.qid, request_id: request.requestId }
        )

        if (reply.status === 'expired') {
          // The wait already ended; the tool's own result says how.
          forgetSettledClarify(request.requestId)

          return false
        }

        if (last) {
          // The last lock resolved the batch server-side, so no response frame is owed.
          forgetServerRequest(request.requestId)
        }

        return true
      } catch (error) {
        notifyError(error, translateNow('assistant.clarify.sendFailed'))
      }
    }

    forgetSettledClarify(request.requestId)
    rollBackTypedAnswer(request, lockedAnswers, question.qid, answer)

    return false
  })

  lockChains.set(request.requestId, sent)

  void sent.then(() => {
    if (lockChains.get(request.requestId) === sent) {
      lockChains.delete(request.requestId)
    }
  })

  return sent
}

/** Undo one failed lock without touching answers locked after it. */
function rollBackTypedAnswer(
  request: ClarifyRequest,
  lockedAnswers: Record<string, null | string>,
  qid: string,
  answer: string
): void {
  const current = $clarifyRequests.get()[keyFor(request.sessionId)]

  // The last answer cleared the card: bring it back while the server still waits.
  const live =
    current?.requestId === request.requestId
      ? current
      : !current && hasOpenServerRequest(request.requestId)
        ? { ...request, lockedAnswers }
        : null

  if (live?.lockedAnswers?.[qid] !== answer) {
    return
  }

  const kept = { ...live.lockedAnswers }
  delete kept[qid]
  setClarifyRequest({ ...live, lockedAnswers: kept })
}
