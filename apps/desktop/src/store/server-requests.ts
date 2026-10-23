import type { ServerRequest } from '@hermes/shared'

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
const open = new Map<string, ServerRequest>()

export function rememberServerRequest(request: ServerRequest): void {
  open.set(request.id, request)
}

export function forgetServerRequest(id: string): void {
  open.delete(id)
}

/** Answer request `id`. False when nothing is open under that id (expired / already answered). */
export function respondToServerRequest(id: string | undefined, result: Record<string, unknown>): boolean {
  const request = id ? open.get(id) : undefined

  if (!request) {
    return false
  }

  open.delete(id!)
  request.respond(result)

  return true
}

export function hasOpenServerRequest(id: string): boolean {
  return open.has(id)
}

export function resetServerRequestsForTests(): void {
  open.clear()
}
