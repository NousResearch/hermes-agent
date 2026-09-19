import type {
  ConnectionOperationStatus,
  ConnectionOperationTarget,
  ConnectionRequestPayload,
  ConnectionSettleReason,
  ConnectionTargetAction,
  ConnectionTargetKind,
  ConnectionTargetState,
  ConnectionUpdatePayload
} from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'

export type { ConnectionSettleReason, ConnectionTargetAction, ConnectionTargetKind, ConnectionTargetState }

/** One target of the operation as the renderer knows it. State comes only from the backend
 *  (`connection.request`, `connectors.operation.status`, `connection.update`); the card never sets it. */
export interface ConnectionTarget {
  name: string
  kind: ConnectionTargetKind
  action: ConnectionTargetAction
  state: ConnectionTargetState
  detail: string
  connectUrl: null | string
  tools: string[]
}

/** The session's connection operation. `deadlineAt`, `opId`, `targets[].state`, `settled` and
 *  `settledBy` are backend-owned; the renderer holds a cache and drives it through `connection.respond`. */
export interface ConnectionRequest {
  /** The model's tool call that opened the operation. The card lives on that row and no other. */
  toolCallId: string
  opId: string
  /** Unix seconds; backend-owned. */
  deadlineAt: number
  targets: ConnectionTarget[]
  settled: boolean
  settledBy: ConnectionSettleReason | null
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
}

/** Answers the card may give for one target. Anything else the backend refuses (4002). */
export type ConnectionTargetOutcome =
  | { name: string; status: 'skipped' }
  | { name: string; status: 'connected'; tools?: string[] }
  | { name: string; status: 'initiated' }
  | { name: string; status: 'failed'; detail?: string }

export interface ConnectionOutcome {
  targets?: ConnectionTargetOutcome[]
  /** `continue` ends the operation now with unresolved targets stamped `not_connected`. */
  settled_by?: 'continue'
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $connectionRequests = atom<Record<string, ConnectionRequest>>({})

export const sessionConnectionRequest = (sessionId: string | null) =>
  computed($connectionRequests, requests => requests[keyFor(sessionId)] ?? null)

const TARGET_STATES: readonly ConnectionTargetState[] = [
  'connected',
  'expired',
  'failed',
  'initiated',
  'not_connected',
  'pending',
  'skipped',
  'unavailable'
]

const ACTIONS: readonly ConnectionTargetAction[] = ['authorize', 'connect', 'enable', 'install', 'reconnect']
const SETTLE_REASONS: readonly ConnectionSettleReason[] = [
  'all_resolved',
  'continue',
  'deadline',
  'interrupt',
  'unavailable'
]

// The wire carries these as typed literals already; the lookups defend against a backend a version ahead.
const oneOf =
  <T extends string>(allowed: readonly T[]) =>
  (value: null | string | undefined): T | undefined =>
    allowed.find(candidate => candidate === value)

const targetState = oneOf(TARGET_STATES)
const targetAction = oneOf(ACTIONS)
const settleReason = oneOf(SETTLE_REASONS)

function parseTarget(entry: ConnectionOperationTarget): ConnectionTarget | null {
  const name = entry.name.trim()

  if (!name) {
    return null
  }

  return {
    action: targetAction(entry.action) ?? 'install',
    connectUrl: entry.connect_url ?? null,
    detail: entry.detail ?? '',
    kind: entry.kind === 'connector' ? 'connector' : 'mcp',
    name,
    state: targetState(entry.state) ?? 'pending',
    tools: entry.tools ?? []
  }
}

/** Parse a `connection.request` event or the `pending_connection` resume field. Null when the payload
 *  carries no usable operation (no op id, no deadline, no targets). */
export function normalizeConnectionRequest(
  payload: ConnectionRequestPayload | null | undefined,
  sessionId: string | null
): ConnectionRequest | null {
  if (!payload) {
    return null
  }

  const targets = payload.targets.map(parseTarget).filter((target): target is ConnectionTarget => target !== null)

  if (!payload.op_id || !payload.tool_call_id || !(payload.deadline_at > 0) || targets.length === 0) {
    return null
  }

  return {
    deadlineAt: payload.deadline_at,
    opId: payload.op_id,
    receivedAt: Date.now() / 1000,
    sessionId,
    settled: false,
    settledBy: null,
    targets,
    toolCallId: payload.tool_call_id
  }
}

/** Overlay the authoritative `connectors.operation.status` snapshot on the cached request. */
export function applyOperationStatus(request: ConnectionRequest, status: ConnectionOperationStatus): ConnectionRequest {
  if (status.op_id !== request.opId) {
    return request
  }

  const byName = new Map(status.targets.map(target => [target.name, target] as const))

  const targets = request.targets.map(target => {
    const live: ConnectionOperationTarget | undefined = byName.get(target.name)

    return live ? mergeLiveTarget(target, live) : target
  })

  const settledBy = settleReason(status.settled_by) ?? null

  // Same reference on a no-op so subscribers do not re-render for an identical frame.
  const unchanged =
    request.deadlineAt === status.deadline_at &&
    request.settled === status.settled &&
    request.settledBy === settledBy &&
    targets.every((target, index) => target === request.targets[index])

  return unchanged
    ? request
    : { ...request, deadlineAt: status.deadline_at, settled: status.settled, settledBy, targets }
}

function mergeLiveTarget(target: ConnectionTarget, live: ConnectionOperationTarget): ConnectionTarget {
  const next: ConnectionTarget = {
    ...target,
    connectUrl: live.connect_url ?? target.connectUrl,
    detail: live.detail ?? target.detail,
    state: live.state,
    tools: live.tools ?? target.tools
  }

  const same =
    next.connectUrl === target.connectUrl &&
    next.detail === target.detail &&
    next.state === target.state &&
    next.tools.length === target.tools.length &&
    next.tools.every((tool, index) => tool === target.tools[index])

  return same ? target : next
}

/** Apply one `connection.update` frame. Every frame carries the operation's full target snapshot, so
 *  the store overlays it; frames for another operation or for a settled request are ignored. */
export function applyConnectionUpdate(request: ConnectionRequest, update: ConnectionUpdatePayload): ConnectionRequest {
  if (update.op_id !== request.opId || request.settled) {
    return request
  }

  return applyOperationStatus(request, update)
}

export function setConnectionRequest(request: ConnectionRequest): void {
  $connectionRequests.set({ ...$connectionRequests.get(), [keyFor(request.sessionId)]: request })
}

export function updateConnectionRequest(sessionId: string | null, update: ConnectionUpdatePayload): void {
  const current = $connectionRequests.get()[keyFor(sessionId)]

  if (!current) {
    return
  }

  const next = applyConnectionUpdate(current, update)

  if (next !== current) {
    setConnectionRequest(next)
  }
}

export function clearConnectionRequest(opId?: string, sessionId?: string | null): void {
  const requests = $connectionRequests.get()

  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (opId && current.opId !== opId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $connectionRequests.set(next)

    return
  }

  const kept = Object.entries(requests).filter(([, value]) => opId && value.opId !== opId)

  if (kept.length !== Object.keys(requests).length) {
    $connectionRequests.set(Object.fromEntries(kept))
  }
}

/** The composer's Enter handler reads this without subscribing. */
export const hasConnectionRequest = (sessionId: string | null | undefined): boolean => {
  const request = $connectionRequests.get()[keyFor(sessionId)]

  return Boolean(request && !request.settled)
}

/** Drive the operation. The entry stays in the store: the backend answers with `connection.update`
 *  and the card re-renders from that; only settlement removes it. */
export async function respondToConnectionRequest(
  request: ConnectionRequest,
  outcome: ConnectionOutcome
): Promise<boolean> {
  const current = $connectionRequests.get()[keyFor(request.sessionId)]

  if (!current || current.opId !== request.opId || current.settled) {
    return false
  }

  await $gateway.get()?.request('connection.respond', {
    op_id: request.opId,
    result: outcome,
    session_id: request.sessionId
  })

  return true
}

/** Not now on one target. */
export const skipConnectionTarget = (request: ConnectionRequest, name: string): Promise<boolean> =>
  respondToConnectionRequest(request, { targets: [{ name, status: 'skipped' }] })

/** Continue: end the operation now with whatever is unresolved. */
export const continueConnectionRequest = (request: ConnectionRequest): Promise<boolean> =>
  respondToConnectionRequest(request, { settled_by: 'continue' })

// Typing a message while the card is open ends the operation, otherwise the typed message waits behind
// the blocked tool until the deadline.
export async function skipConnectionRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $connectionRequests.get()[keyFor(sessionId)]

  if (!request || request.settled) {
    return false
  }

  try {
    await continueConnectionRequest(request)
  } catch {
    // A failed skip must not block the message; the tool settles at its deadline.
  }

  return true
}
