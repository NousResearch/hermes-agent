import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
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
