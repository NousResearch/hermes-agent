import type { SetupChooseKind, SetupChooseOption } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { hasOpenServerRequest, respondToServerRequest } from './server-requests'
import { $activeSessionId } from './session'

export interface ClarifyQuestion {
  qid: string
  question: string
  choices: string[] | null
  multiSelect: boolean
}

export interface SetupChooseSpec {
  kind: SetupChooseKind
  options: SetupChooseOption[] | null
  multiSelect: boolean
  /** Row ids the card starts with picked (the backend fills them from the setup facts). */
  preselected: string[]
}

export interface ClarifyRequest {
  requestId: string
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
  questions: ClarifyQuestion[]
  /** Answers already locked server-side (reconnect replay): qid → answer, null = skipped. */
  lockedAnswers?: Record<string, null | string>
  setup?: SetupChooseSpec
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

export const SETUP_CHOOSE_QID = 'setup_choose'

const SETUP_CHOOSE_KINDS = new Set<unknown>([
  'accent',
  'connectors',
  'fork',
  'layout',
  'machine_use',
  'plugins',
  'question',
  'theme',
  'tour'
])

export function normalizeSetupChoose(
  params: Record<string, unknown>
): Pick<ClarifyRequest, 'questions' | 'setup'> | null {
  const question = typeof params.question === 'string' ? params.question.trim() : ''

  if (!question || !SETUP_CHOOSE_KINDS.has(params.kind)) {
    return null
  }

  const options =
    Array.isArray(params.options) && params.options.length > 0 ? (params.options as SetupChooseOption[]) : null

  const multiSelect = params.multi_select === true

  const preselected = Array.isArray(params.preselected)
    ? params.preselected.filter((id): id is string => typeof id === 'string')
    : []

  return {
    questions: [
      {
        choices: options ? options.map(option => option.label) : null,
        multiSelect: multiSelect && options !== null,
        qid: SETUP_CHOOSE_QID,
        question
      }
    ],
    setup: { kind: params.kind as SetupChooseKind, multiSelect, options, preselected }
  }
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

interface SetupChooseStage {
  /** Persist a confirmed look; clearing or skipping the card only reverts its preview. */
  commit?: (picked: string[]) => void
  draft: string
  /**
   * The name each row of the card shows, by id: a typed answer still names the rows staged with it, and
   * composer text is matched against the rows of a card whose list the app owns.
   */
  labels: Record<string, string>
  picked: string[]
  /** The card's own pick of a row: it applies the row's look the same way a click does. */
  preview: ((id: string) => void) | null
  revert: (() => void) | null
}

export const EMPTY_SETUP_STAGE: SetupChooseStage = { draft: '', labels: {}, picked: [], preview: null, revert: null }

/**
 * A card's answer: the picked ids, and the names the user saw for them when any id is a row. The backend
 * hands both to the model, which would otherwise guess a name from an id such as `#8a2be2`.
 */
export function setupChooseAnswer(
  picked: string | string[],
  labels: Record<string, string>
): { label?: string | string[]; picked: string | string[] } {
  const ids = Array.isArray(picked) ? picked : [picked]

  if (!ids.some(id => Object.hasOwn(labels, id))) {
    return { picked }
  }

  const named = ids.map(id => (Object.hasOwn(labels, id) ? labels[id] : id))

  return { label: Array.isArray(picked) ? named : named[0], picked }
}

export const $setupChooseStages = atom<Record<string, SetupChooseStage>>({})

export const setupChooseStage = (requestId: string): SetupChooseStage =>
  $setupChooseStages.get()[requestId] ?? EMPTY_SETUP_STAGE

export function stageSetupChoose(requestId: string, patch: Partial<SetupChooseStage>): void {
  const current = setupChooseStage(requestId)

  // A no-op patch keeps the atom reference so subscribers do not rerender.
  const unchanged = Object.entries(patch).every(([key, value]) =>
    // SAFETY: the entries of a Partial<SetupChooseStage> carry only SetupChooseStage keys.
    Object.is(current[key as keyof SetupChooseStage], value)
  )

  if (unchanged && requestId in $setupChooseStages.get()) {
    return
  }

  $setupChooseStages.set({ ...$setupChooseStages.get(), [requestId]: { ...current, ...patch } })
}

function clearSetupChoose(requestId: string): void {
  const next = { ...$setupChooseStages.get() }
  delete next[requestId]
  $setupChooseStages.set(next)
}

export function commitSetupChoose(requestId: string): void {
  const stage = setupChooseStage(requestId)
  clearSetupChoose(requestId)
  stage.commit?.(stage.picked)
}

$clarifyRequests.listen(requests => {
  const live = new Set(Object.values(requests).map(request => request.requestId))

  for (const [requestId, stage] of Object.entries($setupChooseStages.get())) {
    if (!live.has(requestId)) {
      clearSetupChoose(requestId)
      stage.revert?.()
    }
  }
})

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

  settleClarify(
    request,
    request.setup
      ? { outcome: 'cancelled', picked: null }
      : {
          outcome: 'cancelled',
          // Answers locked server-side before a reconnect stand: the server merges them into its result too.
          responses: request.questions.map(question => {
            const locked = request.lockedAnswers?.[question.qid]

            return locked
              ? { question: question.question, status: 'answered', user_response: lockedAnswer(question, locked) }
              : { question: question.question, status: 'unanswered', user_response: null }
          })
        }
  )
}

export async function skipClarifyRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  if (!request) {
    return false
  }

  skipClarify(request)

  return true
}

/**
 * Answer the setup card parked on `sessionId` with text the user typed in the
 * composer: the setup turn reads typed words as the card's answer. False when
 * no setup card is parked (an ordinary clarify card included) or its request is
 * gone; the caller then skips any card and sends the words as a message.
 */
export function answerSetupCard(sessionId: string | null | undefined, text: string): boolean {
  const request = $clarifyRequests.get()[keyFor(sessionId)]

  return request?.setup ? answerSetupChoose(request, request.setup, text) : false
}

/** The row typed text names, by id or label (any case). */
function matchSetupRow(labels: Record<string, string>, text: string): string | undefined {
  const typed = text.trim().toLowerCase()

  return Object.keys(labels).find(id => id.toLowerCase() === typed || labels[id].trim().toLowerCase() === typed)
}

function answerSetupChoose(request: ClarifyRequest, setup: SetupChooseSpec, text: string): boolean {
  const stage = setupChooseStage(request.requestId)
  // The backend fills its own rows into the request; the app's lists are known to the card that drew them.
  const labels = setup.options ? Object.fromEntries(setup.options.map(row => [row.id, row.label])) : stage.labels
  const freeText = setup.kind === 'question' && !setup.options
  const id = freeText ? text : matchSetupRow(labels, text)

  if (id === undefined) {
    // Words that name no row are not a pick: the model reads them and asks again.
    if (!respondToServerRequest(request.requestId, { said: text })) {
      return false
    }

    clearClarifyRequest(request.requestId, request.sessionId)
    settleClarify(request, { outcome: 'typed', picked: null, said: text })

    return true
  }

  // A multi-select picker keeps the rows already staged on the card, the
  // same as its own Confirm: the typed row is one more pick.
  const answer = setupChooseAnswer(
    setup.multiSelect ? [...new Set([...stage.picked, id])] : id,
    freeText ? stage.labels : { ...stage.labels, ...labels }
  )

  if (!respondToServerRequest(request.requestId, answer)) {
    return false
  }

  // A typed row is picked as if clicked, so its look stays once the request clears.
  const staged = setup.multiSelect || stage.picked.includes(id)

  if (!staged) {
    stage.preview?.(id)
  }

  if (staged || stage.preview) {
    commitSetupChoose(request.requestId)
  }

  clearClarifyRequest(request.requestId, request.sessionId)
  settleClarify(request, { outcome: 'submitted', ...answer })

  return true
}
