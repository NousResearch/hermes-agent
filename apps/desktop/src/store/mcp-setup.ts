import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'

/**
 * Pending `mcp.setup.request`s — the desktop half of the `setup_mcp` tool's
 * blocking bridge (tools/setup_mcp_tool.py). Mirrors the clarify store:
 * keyed by the runtime session id that raised the request so a background
 * session can park its card while the user looks at another chat, and the
 * inline McpSetupTool reads its own session's entry.
 */
export interface McpSetupRequest {
  requestId: string
  /** First connector — kept alongside `servers` for anything reading a scalar. */
  server: string
  /** Every connector the card should offer, in the order the agent asked. */
  servers: string[]
  /** `connect` lets the card resolve the step from each connector's real
   *  state; the others are the agent naming one specific step. */
  action: 'authorize' | 'connect' | 'enable' | 'install'
  /** Agent-supplied one-liner: why these help right now. */
  reason: string
  /** Prerequisites the agent wrote for a connector that has none of its own.
   *  A reviewed connector's manifest steps always win — those were checked;
   *  these are the model's best account of an unreviewed setup, which still
   *  beats the same account buried in a chat message above the card. */
  steps: string[]
  sessionId: string | null
}

/** One connector's result inside a card's answer. */
export interface McpConnectorOutcome {
  server: string
  status: 'connected' | 'declined' | 'error'
  detail?: string
  /** Tool names the connector brought in, when the flow learned them. */
  tools?: string[]
  /** The failure was a refusal, not a fault: wrong key, expired grant, or a
   *  grant that never covered the tools. Asking again is the fix, so the card
   *  offers access rather than a retry. */
  needsAuth?: boolean
}

/** The card's answer, serialized back through `mcp.setup.respond`.
 *  `status` is the aggregate; `connectors` is the per-row truth. */
export interface McpSetupOutcome {
  status: 'connected' | 'declined' | 'error' | 'partial'
  server: string
  connectors: McpConnectorOutcome[]
  detail?: string
}

/**
 * Fold each connector card's result into the one answer the tool gets.
 *
 * The aggregate is derived, never authoritative: `connectors` is the truth and
 * `status` only summarizes it, so two of three connecting reports `partial`
 * with both successes intact rather than collapsing to a single verdict.
 *
 * A connector with no result was never connected — its card was dismissed, or
 * the user walked away from the whole set — so it reports `declined`.
 */
export function buildSetupOutcome(input: {
  names: string[]
  results: Record<string, McpConnectorOutcome>
  server: string
}): McpSetupOutcome {
  const { names, results, server } = input

  const connectors: McpConnectorOutcome[] = names.map(name => results[name] ?? { server: name, status: 'declined' })

  const connected = connectors.filter(connector => connector.status === 'connected').length
  const failed = connectors.some(connector => connector.status === 'error')

  return {
    connectors,
    server,
    status: connected === 0 ? (failed ? 'error' : 'declined') : failed ? 'partial' : 'connected'
  }
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $mcpSetupRequests = atom<Record<string, McpSetupRequest>>({})

/** The setup request for one specific session — the transcript card reads
 *  this fixed-key view, same shape as `sessionClarifyRequest`. */
export const sessionMcpSetupRequest = (sessionId: string | null) =>
  computed($mcpSetupRequests, requests => requests[keyFor(sessionId)] ?? null)

export function setMcpSetupRequest(request: McpSetupRequest): void {
  $mcpSetupRequests.set({ ...$mcpSetupRequests.get(), [keyFor(request.sessionId)]: request })
}

export function clearMcpSetupRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $mcpSetupRequests.get()

  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $mcpSetupRequests.set(next)

    return
  }

  const next: Record<string, McpSetupRequest> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      changed = true
    }
  }

  if (changed) {
    $mcpSetupRequests.set(next)
  }
}

/** Whether `sessionId` has a setup card pending right now (imperative read —
 *  the composer checks this on Enter, not on every render). */
export const hasMcpSetupRequest = (sessionId: string | null | undefined): boolean =>
  Boolean($mcpSetupRequests.get()[keyFor(sessionId)])

/**
 * Answer `sessionId`'s pending setup card as declined and drop it locally,
 * resolving to whether there was one to skip.
 *
 * The composer uses this when the user types a real message instead of acting
 * on the card: setup_mcp blocks the agent inside its tool batch, so leaving
 * the card unanswered would park the follow-up until the 10-minute timeout —
 * the message looks sent and nothing happens. Typing IS the answer "not now":
 * decline so the tool returns, then route the words normally.
 *
 * Mirrors skipClarifyRequest; mcp.setup.respond is allow_expired, so racing
 * the timeout is harmless.
 */
export async function skipMcpSetupRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $mcpSetupRequests.get()[keyFor(sessionId)]

  if (!request) {
    return false
  }

  // Clear first: the answer is already decided, and an in-flight RPC must not
  // leave a live card the user can answer a second time.
  clearMcpSetupRequest(request.requestId, request.sessionId)

  try {
    await $gateway.get()?.request('mcp.setup.respond', {
      request_id: request.requestId,
      result: JSON.stringify({
        connectors: request.servers.map(server => ({ server, status: 'declined' })),
        server: request.server,
        status: 'declined'
      })
    })
  } catch {
    // The tool times out on its own; a failed skip must never swallow the
    // message the user is actually sending.
  }

  return true
}
