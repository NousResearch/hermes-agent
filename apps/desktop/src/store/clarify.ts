import { atom } from 'nanostores'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

// Holds the request_id (and metadata) for the most recent in-flight
// clarify call. The inline ClarifyTool component (rendered inside the
// assistant message stream) reads this to know which request_id to send
// back over `clarify.respond`.
export const $clarifyRequest = atom<ClarifyRequest | null>(null)

export function setClarifyRequest(request: ClarifyRequest): void {
  $clarifyRequest.set(request)
}

export function clearClarifyRequest(requestId?: string): void {
  const current = $clarifyRequest.get()

  if (!current) {
    return
  }

  if (requestId && current.requestId !== requestId) {
    return
  }

  $clarifyRequest.set(null)
}
