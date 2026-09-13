import { isRecord } from '@assistant-ui/core/internal'
import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'

/** Pending `connection.request`s, keyed by runtime session id (mirrors the clarify store).
 *  The backend owns `opId`, targets and `deadlineAt`; the renderer never recomputes them. */
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

/** One target's answer. `declined` = Not now; `error` = recoverable failure. */
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

/** One session's pending request (same shape as `sessionClarifyRequest`). */
export const sessionConnectionRequest = (sessionId: string | null) =>
  computed($connectionRequests, requests => requests[keyFor(sessionId)] ?? null)

const ACTIONS: readonly ConnectionAction[] = ['install', 'enable', 'authorize']

/** The wire shape of `connection.request` and the `pending_connection` resume field. */
export interface ConnectionRequestWire {
  request_id?: string
  op_id?: string
  deadline_at?: number
  reason?: string
  targets?: unknown
}

const str = (value: string | undefined): string => value ?? ''

/** Validate a wire payload. Null when it carries no usable operation (no request id, no targets). */
export function normalizeConnectionRequest(
  payload: ConnectionRequestWire | null | undefined,
  sessionId: string | null
): ConnectionRequest | null {
  if (!payload) {
    return null
  }

  const requestId = str(payload.request_id)
  const opId = str(payload.op_id)
  const deadlineAt = payload.deadline_at && payload.deadline_at > 0 ? payload.deadline_at : 0
  const rawTargets = Array.isArray(payload.targets) ? payload.targets : []

  const targets: ConnectionTarget[] = rawTargets.flatMap(entry => {
    if (!isRecord(entry)) {
      return []
    }

    // SAFETY: isRecord narrowed to an object; each field is re-checked against its allowed values below.
    const t = entry as { action?: unknown; kind?: unknown; name?: unknown }
    const name = String(t.name ?? '').trim()
    const action = ACTIONS.find(a => a === t.action) ?? 'install'

    return name && t.name === name.trim() ? [{ action, kind: t.kind === 'connector' ? 'connector' : 'mcp', name }] : []
  })

  if (!requestId || !opId || !deadlineAt || targets.length === 0) {
    return null
  }

  return {
    deadlineAt,
    opId,
    reason: str(payload.reason),
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

/** Imperative read for the composer's Enter handler. */
export const hasConnectionRequest = (sessionId: string | null | undefined): boolean =>
  Boolean($connectionRequests.get()[keyFor(sessionId)])

/** Send the card's answer. Clears the entry first so the card cannot be answered twice;
 *  false when the request is already gone. `connection.respond` tolerates a late answer. */
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

/** Typing a message while the card is up declines every target (mirrors skipClarifyRequest);
 *  otherwise the follow-up would park until the deadline. */
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
    // A failed skip must not swallow the message being sent; the tool settles on its deadline.
  }

  return true
}
