import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
  answerDraft?: string
  selectedChoice?: string | null
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

export function setClarifyRequest(request: ClarifyRequest): void {
  const key = keyFor(request.sessionId)
  const current = $clarifyRequests.get()[key]
  const samePrompt = current && (current.requestId === request.requestId || current.question === request.question)
  const selectedChoice =
    current?.selectedChoice && request.choices?.includes(current.selectedChoice) ? current.selectedChoice : null

  $clarifyRequests.set({
    ...$clarifyRequests.get(),
    [key]: samePrompt
      ? {
          ...request,
          answerDraft: current.answerDraft,
          selectedChoice
        }
      : request
  })
}

export function updateClarifyAnswerDraft(requestId: string, sessionId: string | null | undefined, answerDraft: string) {
  const key = keyFor(sessionId)
  const requests = $clarifyRequests.get()
  const current = requests[key]

  if (!current || current.requestId !== requestId) {
    return
  }

  $clarifyRequests.set({
    ...requests,
    [key]: {
      ...current,
      answerDraft,
      selectedChoice: answerDraft.trim() ? null : current.selectedChoice ?? null
    }
  })
}

export function updateClarifySelectedChoice(
  requestId: string,
  sessionId: string | null | undefined,
  selectedChoice: string | null
) {
  const key = keyFor(sessionId)
  const requests = $clarifyRequests.get()
  const current = requests[key]

  if (!current || current.requestId !== requestId) {
    return
  }

  $clarifyRequests.set({
    ...requests,
    [key]: {
      ...current,
      answerDraft: selectedChoice ? '' : current.answerDraft,
      selectedChoice
    }
  })
}

export function clearClarifyRequest(requestId?: string, sessionId?: string | null): ClarifyRequest[] {
  const requests = $clarifyRequests.get()

  // Targeted clear when the caller knows the session (the common path from the
  // inline ClarifyTool answering its own request).
  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return []
    }

    const next = { ...requests }
    delete next[key]
    $clarifyRequests.set(next)

    return [current]
  }

  // Fallback with no session hint: drop every entry matching the request id
  // (or clear all when none is given).
  const next: Record<string, ClarifyRequest> = {}
  const cleared: ClarifyRequest[] = []
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      cleared.push(value)
      changed = true
    }
  }

  if (changed) {
    $clarifyRequests.set(next)
  }

  return cleared
}
