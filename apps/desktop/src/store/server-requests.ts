import type { AnyServerRequest, ServerRequest, ServerRequestMap } from '@hermes/shared'

/**
 * Live server→client requests (`tui_gateway/server_requests.py`) keyed by
 * request id: clarify / approval / sudo / secret / vault / MCP-setup cards.
 *
 * The per-session prompt stores keep only the id; a card answers through
 * `respondToServerRequest`, which routes the response frame back over the
 * socket the request arrived on — the owner backend by construction, so no
 * owner-route lookup is needed (#91684's whole class disappears: the answer
 * cannot land on the wrong backend because it is a JSON-RPC response, not a
 * new call). A request re-delivered after a reconnect (`open_requests`)
 * carries the same id and replaces the entry, so the still-visible card
 * answers the new generation.
 */
const open = new Map<string, AnyServerRequest>()

export function rememberServerRequest(request: AnyServerRequest): void {
  open.set(request.id, request)
}

export function forgetServerRequest(id: string): void {
  open.delete(id)
}

/**
 * Answer the `method` request open under `id` with its typed reply. False when
 * nothing of that kind is open under the id (expired / already answered).
 */
export function respondToServerRequest<M extends keyof ServerRequestMap>(
  method: M,
  id: string | undefined,
  result: ServerRequestMap[M]['result']
): boolean {
  const request = id === undefined ? undefined : open.get(id)

  if (!request || request.method !== method) {
    return false
  }

  open.delete(request.id)

  // `respond` is a method signature, so the matched member widens to the whole
  // map without a cast; the method check above is what makes that exact.
  const matched: ServerRequest<keyof ServerRequestMap> = request
  matched.respond(result)

  return true
}

/** Withdraw request `id` with a JSON-RPC error: the backend settles it as unanswered. */
export function cancelServerRequest(id: string): boolean {
  const request = open.get(id)

  if (!request) {
    return false
  }

  open.delete(id)
  request.fail({ message: 'cancelled by the user' })

  return true
}

export function hasOpenServerRequest(id: string): boolean {
  return open.has(id)
}

export function resetServerRequestsForTests(): void {
  open.clear()
}
