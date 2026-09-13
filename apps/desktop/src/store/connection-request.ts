import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'

/**
 * Pending `connection.request`s — the desktop half of the `manage_connections`
 * tool's blocking bridge for local MCP targets (tools/connections_tool_mcp.py).
 * Mirrors the clarify store: keyed by the runtime session id that raised the
 * request so a background session can park its card while the user looks at
 * another chat, and the inline card reads its own session's entry.
 *
 * The backend owns the operation: `opId`, the target list and `deadlineAt`
 * are fixed when the tool call starts. Navigation, remount and desktop
 * restart never recompute the deadline; `session.resume` replays this same
 * payload as `pending_connection`.
 */
export type ConnectionTargetKind = 'connector' | 'mcp'
export type ConnectionAction = 'authorize' | 'enable' | 'install'

export interface ConnectionTarget {
  name: string
  kind: ConnectionTargetKind
  action: ConnectionAction
}

export interface ConnectionRequest {
  requestId: string
  opId: string
  /** Unix seconds, server-owned. */
  deadlineAt: number
  /** Agent-supplied one-liner: why this helps right now. */
  reason: string
  targets: ConnectionTarget[]
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
}

/** One target's answer. `declined` is the user's Not now; `error` is a
 *  recoverable failure the tool records as `failed`. */
export type ConnectionTargetStatus = 'authorized' | 'declined' | 'enabled' | 'error' | 'installed'

export interface ConnectionTargetOutcome {
  name: string
  status: ConnectionTargetStatus
  detail?: string
  /** Tool names now available (OAuth flows report them). */
  tools?: string[]
}

/** The card's answer, serialized back through `connection.respond`. */
export interface ConnectionOutcome {
  targets: ConnectionTargetOutcome[]
  /** How the card asked to settle; omitted = let the backend decide from the target states. */
  settled_by?: 'all_resolved' | 'continue'
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $connectionRequests = atom<Record<string, ConnectionRequest>>({})

/** The pending request for one specific session — the transcript card reads
 *  this fixed-key view, same shape as `sessionClarifyRequest`. */
export const sessionConnectionRequest = (sessionId: string | null) =>
  computed($connectionRequests, requests => requests[keyFor(sessionId)] ?? null)

const ACTIONS: readonly ConnectionAction[] = ['install', 'enable', 'authorize']

/** Validate a wire `connection.request` / `pending_connection` payload. Null
 *  when it carries no usable operation (no request id, no targets). */
export function normalizeConnectionRequest(payload: unknown, sessionId: string | null): ConnectionRequest | null {
  if (typeof payload !== 'object' || payload === null) {
    return null
  }

  const row = payload as Record<string, unknown>
  const requestId = typeof row.request_id === 'string' ? row.request_id : ''
  const opId = typeof row.op_id === 'string' ? row.op_id : ''
  const deadlineAt = typeof row.deadline_at === 'number' && row.deadline_at > 0 ? row.deadline_at : 0
  const rawTargets = Array.isArray(row.targets) ? row.targets : []

  const targets: ConnectionTarget[] = rawTargets.flatMap(entry => {
    if (typeof entry !== 'object' || entry === null) {
      return []
    }

    const t = entry as Record<string, unknown>
    const name = typeof t.name === 'string' ? t.name.trim() : ''
    const action = ACTIONS.find(a => a === t.action) ?? 'install'

    return name ? [{ action, kind: t.kind === 'connector' ? 'connector' : 'mcp', name }] : []
  })

  if (!requestId || !opId || !deadlineAt || targets.length === 0) {
    return null
  }

  return {
    deadlineAt,
    opId,
    reason: typeof row.reason === 'string' ? row.reason : '',
    receivedAt: Date.now() / 1000,
    requestId,
    sessionId,
    targets
  }
}

export function setConnectionRequest(request: ConnectionRequest): void {
  $connectionRequests.set({ ...$connectionRequests.get(), [keyFor(request.sessionId)]: request })
}

export function clearConnectionRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $connectionRequests.get()

  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $connectionRequests.set(next)

    return
  }

  const next: Record<string, ConnectionRequest> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      changed = true
    }
  }

  if (changed) {
    $connectionRequests.set(next)
  }
}

/** Whether `sessionId` has a connection card pending right now (imperative
 *  read — the composer checks this on Enter, not on every render). */
export const hasConnectionRequest = (sessionId: string | null | undefined): boolean =>
  Boolean($connectionRequests.get()[keyFor(sessionId)])

/** Send the card's answer. Clears the local entry FIRST so an in-flight RPC
 *  can never leave a live card the user can answer a second time. Resolves
 *  to false when this session's request is already gone (cancel racing
 *  completion, or a stale card). `connection.respond` is allow_expired, so
 *  racing the backend deadline is harmless. */
export async function respondToConnectionRequest(request: ConnectionRequest, outcome: ConnectionOutcome): Promise<boolean> {
  const current = $connectionRequests.get()[keyFor(request.sessionId)]

  if (!current || current.requestId !== request.requestId) {
    return false
  }

  clearConnectionRequest(request.requestId, request.sessionId)

  await $gateway.get()?.request('connection.respond', {
    request_id: request.requestId,
    result: JSON.stringify(outcome)
  })

  return true
}

/**
 * Answer `sessionId`'s pending card as declined on every target and drop it
 * locally, resolving to whether there was one to skip.
 *
 * The composer uses this when the user types a real message instead of acting
 * on the card: the tool blocks the agent inside its tool batch, so leaving the
 * card unanswered would park the follow-up until the deadline. Typing IS the
 * answer "not now" for every target. Mirrors skipClarifyRequest.
 */
export async function skipConnectionRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $connectionRequests.get()[keyFor(sessionId)]

  if (!request) {
    return false
  }

  try {
    await respondToConnectionRequest(request, {
      settled_by: 'all_resolved',
      targets: request.targets.map(target => ({ name: target.name, status: 'declined' }))
    })
  } catch {
    // The tool settles on its own deadline; a failed skip must never swallow
    // the message the user is actually sending.
  }

  return true
}
