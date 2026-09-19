import type {
  ConnectionLeg,
  ConnectionLegAction,
  ConnectionLegEnvField,
  ConnectionLegKind,
  ConnectionLegState,
  ConnectionOperationStatus,
  ConnectionRequestPayload,
  ConnectionSettleReason,
  ConnectionUpdatePayload
} from '@hermes/shared'
import { atom, computed } from 'nanostores'

import type { ConnectorCardField } from '@/components/ui/connector-card'

import { $gateway } from './gateway'

/** The backend sends ``prompt`` as null when the catalog entry has none; the card takes an absent one. */
const envFields = (fields: ConnectionLegEnvField[] | null | undefined): ConnectionEnvField[] =>
  (fields ?? []).map(({ default: defaultValue, name, prompt, required, secret }) => ({
    default: defaultValue ?? null,
    name,
    prompt: prompt ?? undefined,
    required,
    secret
  }))

export type {
  ConnectionLegAction,
  ConnectionLegEnvField,
  ConnectionLegKind,
  ConnectionLegState,
  ConnectionSettleReason
}

/** One leg of the operation as the renderer knows it. State comes only from the backend
 *  (`connection.request`, `connectors.operation.status`, `connection.update`); the card never sets it. */
export interface ConnectionEnvField extends ConnectorCardField {
  default: string | null
  secret: boolean
}

export interface ConnectionRequestLeg {
  name: string
  kind: ConnectionLegKind
  action: ConnectionLegAction
  state: ConnectionLegState
  detail: string
  connectUrl: null | string
  /** The vendor account of a managed leg once a mint named one; empty before that and on MCP legs. */
  connectionId: string
  /** Toolkit metadata on connector legs; empty on an MCP leg. */
  tools: string[]
  /** Credentials an MCP install is still waiting for; empty on every other leg. */
  requiredEnv: ConnectionEnvField[]
}

/** The session's connection operation. `deadlineAt`, `opId`, `legs[].state`, `settled` and
 *  `settledBy` are backend-owned; the renderer holds a cache and drives it through `connection.respond`. */
export interface ConnectionRequest {
  /** The model's tool call that opened the operation. The card lives on that row and no other. */
  toolCallId: string
  opId: string
  /** The sequence of the newest frame this cache holds; an older frame for the same op is dropped. */
  seq: number
  /** Unix seconds; backend-owned. */
  deadlineAt: number
  legs: ConnectionRequestLeg[]
  settled: boolean
  settledBy: ConnectionSettleReason | null
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
}

/** Answers the card may give for one leg: the user said no, or the user consented and the backend
 *  does the work. The card never reports an outcome; only the backend moves a leg. */
export type ConnectionLegOutcome =
  | { name: string; status: 'skipped' }
  | { env?: Record<string, string>; name: string; status: 'approved' }

export interface ConnectionOutcome {
  legs?: ConnectionLegOutcome[]
  /** `continue` ends the operation now with unresolved legs stamped `not_connected`. */
  settled_by?: 'continue'
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $connectionRequests = atom<Record<string, ConnectionRequest>>({})

export const sessionConnectionRequest = (sessionId: string | null) =>
  computed($connectionRequests, requests => requests[keyFor(sessionId)] ?? null)

const LEG_STATES: readonly ConnectionLegState[] = [
  'connected',
  'expired',
  'failed',
  'initiated',
  'not_connected',
  'pending',
  'skipped',
  'unavailable'
]

const ACTIONS: readonly ConnectionLegAction[] = ['authorize', 'connect', 'enable', 'install', 'reconnect']
const SETTLE_REASONS: readonly ConnectionSettleReason[] = ['all_resolved', 'continue', 'deadline', 'interrupt', 'unavailable']

// The wire carries these as typed literals already; the lookups defend against a backend a version ahead.
const oneOf =
  <T extends string>(allowed: readonly T[]) =>
  (value: null | string | undefined): T | undefined =>
    allowed.find(candidate => candidate === value)

const legState = oneOf(LEG_STATES)
const legAction = oneOf(ACTIONS)
const settleReason = oneOf(SETTLE_REASONS)

function parseLeg(entry: ConnectionLeg): ConnectionRequestLeg | null {
  const name = entry.name.trim()

  if (!name) {
    return null
  }

  return {
    action: legAction(entry.action) ?? 'install',
    connectUrl: entry.connect_url ?? null,
    detail: entry.detail ?? '',
    kind: entry.kind === 'connector' ? 'connector' : 'mcp',
    name,
    state: legState(entry.state) ?? 'pending',
    tools: entry.tools ?? [],
    connectionId: entry.connection_id ?? '',
    requiredEnv: envFields(entry.required_env)
  }
}

/** Parse a `connection.request` event or the `pending_connection` resume field. Null when the payload
 *  carries no usable operation (no op id, no deadline, no legs). */
export function normalizeConnectionRequest(
  payload: ConnectionRequestPayload | null | undefined,
  sessionId: string | null
): ConnectionRequest | null {
  if (!payload) {
    return null
  }

  const legs = payload.legs.map(parseLeg).filter((leg): leg is ConnectionRequestLeg => leg !== null)

  if (!payload.op_id || !payload.tool_call_id || !(payload.deadline_at > 0) || legs.length === 0) {
    return null
  }

  return {
    deadlineAt: payload.deadline_at,
    opId: payload.op_id,
    receivedAt: Date.now() / 1000,
    seq: payload.seq,
    sessionId,
    settled: false,
    settledBy: null,
    legs,
    toolCallId: payload.tool_call_id
  }
}

/** Overlay the authoritative `connectors.operation.status` snapshot on the cached request. Frames for
 *  another operation, and frames the operation wrote before the one already applied, change nothing:
 *  the transport can reorder them and an older one would regress a row. */
export function applyOperationStatus(
  request: ConnectionRequest,
  status: ConnectionOperationStatus
): ConnectionRequest {
  if (status.op_id !== request.opId || status.seq <= request.seq) {
    return request
  }

  const byName = new Map(status.legs.map(leg => [leg.name, leg] as const))

  const legs = request.legs.map(leg => {
    const live: ConnectionLeg | undefined = byName.get(leg.name)

    return live ? mergeLiveLeg(leg, live) : leg
  })

  const settledBy = settleReason(status.settled_by) ?? null

  // Same reference on a no-op so subscribers do not re-render for an identical frame.
  const unchanged =
    request.deadlineAt === status.deadline_at &&
    request.seq === status.seq &&
    request.settled === status.settled &&
    request.settledBy === settledBy &&
    legs.every((leg, index) => leg === request.legs[index])

  return unchanged
    ? request
    : { ...request, deadlineAt: status.deadline_at, seq: status.seq, settled: status.settled, settledBy, legs }
}

function mergeLiveLeg(leg: ConnectionRequestLeg, live: ConnectionLeg): ConnectionRequestLeg {
  const next: ConnectionRequestLeg = {
    ...leg,
    connectUrl: live.connect_url ?? leg.connectUrl,
    detail: live.detail ?? leg.detail,
    state: live.state,
    tools: live.tools ?? leg.tools,
    connectionId: live.connection_id ?? leg.connectionId,
    requiredEnv: live.required_env ? envFields(live.required_env) : leg.requiredEnv
  }

  const same =
    next.connectUrl === leg.connectUrl &&
    next.connectionId === leg.connectionId &&
    next.detail === leg.detail &&
    next.state === leg.state &&
    next.tools.length === leg.tools.length &&
    next.tools.every((tool, index) => tool === leg.tools[index]) &&
    sameEnvFields(next.requiredEnv, leg.requiredEnv)

  return same ? leg : next
}

// Every frame carries a fresh array, so identity would churn the row and remount its open inputs.
const sameEnvFields = (next: ConnectionEnvField[], previous: ConnectionEnvField[]): boolean =>
  next.length === previous.length &&
  next.every(
    (field, index) =>
      field.name === previous[index].name &&
      field.prompt === previous[index].prompt &&
      field.default === previous[index].default &&
      field.required === previous[index].required &&
      field.secret === previous[index].secret
  )

/** Apply one `connection.update` frame. Every frame carries the operation's full leg snapshot, so
 *  the store overlays it; frames for another operation or for a settled request are ignored. */
export function applyConnectionUpdate(
  request: ConnectionRequest,
  update: ConnectionUpdatePayload
): ConnectionRequest {
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
export async function respondToConnectionRequest(request: ConnectionRequest, outcome: ConnectionOutcome): Promise<boolean> {
  const current = $connectionRequests.get()[keyFor(request.sessionId)]

  if (!current || current.opId !== request.opId || current.settled || !request.sessionId) {
    return false
  }

  await $gateway.get()?.request('connection.respond', {
    op_id: request.opId,
    owner: { session_id: request.sessionId, type: 'session' },
    result: outcome
  })

  return true
}

/** Not now on one leg. */
export const skipConnectionLeg = (request: ConnectionRequest, name: string): Promise<boolean> =>
  respondToConnectionRequest(request, { legs: [{ name, status: 'skipped' }] })

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
