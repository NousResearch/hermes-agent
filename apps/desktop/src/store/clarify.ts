import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

// Pending clarify requests keyed by the runtime session id that raised them.
// Storing per-session (instead of one shared slot) lets a *background* session
// park its clarify request while the user is looking at a different chat, then
// resolve it once they switch over — without another session clobbering it. We
// store a FIFO queue per session because the Python gateway itself allows more
// than one clarify to be pending in the same session (tools/clarify_gateway.py
// keeps a list per session and resolves oldest-first). A request with no
// session id lands under the empty key.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

type ClarifyQueues = Record<string, ClarifyRequest[]>

function queueFor(requests: ClarifyQueues, sessionId: string | null | undefined): ClarifyRequest[] {
  return requests[keyFor(sessionId)] ?? []
}

export const $clarifyRequests = atom<ClarifyQueues>({})

// The ACTIVE clarify request for the currently-viewed session (oldest pending).
// Both the modal overlay and the inline ClarifyTool only ever care about the
// head of the active session's queue.
export const $clarifyRequest = computed(
  [$clarifyRequests, $activeSessionId],
  (requests, activeId) => queueFor(requests, activeId)[0] ?? null
)

export function hasClarifyRequest(sessionId?: string | null): boolean {
  const requests = $clarifyRequests.get()

  if (sessionId !== undefined) {
    return queueFor(requests, sessionId).length > 0
  }

  return Object.values(requests).some(queue => queue.length > 0)
}

export function setClarifyRequest(request: ClarifyRequest): void {
  const all = $clarifyRequests.get()
  const key = keyFor(request.sessionId)
  const queue = [...queueFor(all, request.sessionId)]
  const existingIndex = queue.findIndex(value => value.requestId === request.requestId)

  if (existingIndex >= 0) {
    queue[existingIndex] = request
  } else {
    queue.push(request)
  }

  $clarifyRequests.set({ ...all, [key]: queue })
}

export function clearClarifyRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $clarifyRequests.get()

  // Targeted clear when the caller knows the session (the common path from the
  // overlay / inline ClarifyTool answering its own request). With no request id
  // we drop the whole session queue (turn ended, timeout, interrupt).
  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = queueFor(requests, sessionId)

    if (!current.length) {
      return
    }

    const nextQueue = requestId ? current.filter(value => value.requestId !== requestId) : []

    if (requestId && nextQueue.length === current.length) {
      return
    }

    const next = { ...requests }

    if (nextQueue.length > 0) {
      next[key] = nextQueue
    } else {
      delete next[key]
    }

    $clarifyRequests.set(next)

    return
  }

  if (!requestId) {
    if (Object.keys(requests).length > 0) {
      $clarifyRequests.set({})
    }

    return
  }

  // Fallback with no session hint: drop matching request ids across every
  // parked session queue.
  const next: ClarifyQueues = {}
  let changed = false

  for (const [key, queue] of Object.entries(requests)) {
    const filtered = queue.filter(value => value.requestId !== requestId)

    if (filtered.length !== queue.length) {
      changed = true
    }

    if (filtered.length > 0) {
      next[key] = filtered
    }
  }

  if (changed) {
    $clarifyRequests.set(next)
  }
}
