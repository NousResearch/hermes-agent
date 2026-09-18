// The only file on the Connectors page that knows the wire.
//
// Every call is typed against the generated contract (`RpcMethods`), goes
// through the app's per-scope request helper, and answers a `ConnectorRpcError`
// carrying the backend's closed `reason`. Nothing above this file catches a raw
// JSON-RPC error, and nothing below it knows what a card looks like.
//
// The page has no chat session, so every operation call names the ACCOUNT owner.
// The session-owned twins of these RPCs belong to the chat connector card and
// are not reachable from here.

import type {
  AccountOwner,
  ConnectionAnswer,
  ConnectorChange,
  RpcMethods,
  ToolsChange
} from '@hermes/shared'
import { JsonRpcGatewayError } from '@hermes/shared'

import type { ProfileScope } from '@/hermes'
import { requestGatewayForAgent } from '@/store/gateway'

/** `tui_gateway/contracts/connectors.py::ConnectorErrorReason`.
 *
 *  The contract generator emits declared params and results; the reason a
 *  handler puts in `error.data` is not one of them, so the closed set is
 *  restated here. A reason added on the backend without being added here costs
 *  only the generic branch — `reason` comes back `undefined` and the caller
 *  treats the failure as "failed", never as a wrong typed state. */
export const CONNECTOR_ERROR_REASONS = [
  'ACCOUNTS_UNAVAILABLE',
  'CATALOG_UNAVAILABLE',
  'CONNECTION_NOT_FOUND',
  'CONNECTORS_UNAVAILABLE',
  'CONNECTOR_NOT_FOUND',
  'CONNECTOR_REQUEST_FAILED',
  'FORBIDDEN_SCOPE',
  'INVALID_ANSWER',
  'INVALID_CONNECTOR_RESPONSE',
  'INVALID_PARAMS',
  'INVALID_POLICY',
  'LINK_STILL_VALID',
  'NEEDS_NOUS_AUTH',
  'NOT_OWNER',
  'POLICY_CONFLICT',
  'POLICY_UNAVAILABLE',
  'REISSUE_REFUSED',
  'TOOLS_UNAVAILABLE',
  'UNKNOWN_OPERATION',
  'UNKNOWN_TARGET',
  'UNSUPPORTED_RUNTIME'
] as const

export type ConnectorErrorReason = (typeof CONNECTOR_ERROR_REASONS)[number]

const REASONS: ReadonlySet<string> = new Set(CONNECTOR_ERROR_REASONS)

/** One failure shape for the whole page. `reason` is the backend's verdict;
 *  `undefined` means the failure never reached a connector handler (transport
 *  down, deadline, a backend that predates the method). */
export class ConnectorRpcError extends Error {
  readonly code: number | undefined
  readonly reason: ConnectorErrorReason | undefined

  constructor(message: string, code?: number, reason?: ConnectorErrorReason) {
    super(message)
    this.name = 'ConnectorRpcError'
    this.code = code
    this.reason = reason
  }
}

/** Read `error.data.reason` off a gateway error. The frame carries `data` all
 *  the way into the renderer on the WebSocket path; a rethrow that lost it
 *  simply yields `undefined` rather than a guessed reason. */
function reasonOf(error: unknown): ConnectorErrorReason | undefined {
  const data = error instanceof JsonRpcGatewayError ? error.data : undefined
  const reason = data && typeof data === 'object' ? (data as { reason?: unknown }).reason : undefined

  return typeof reason === 'string' && REASONS.has(reason) ? (reason as ConnectorErrorReason) : undefined
}

export function asConnectorError(error: unknown): ConnectorRpcError {
  if (error instanceof ConnectorRpcError) {
    return error
  }

  const message = error instanceof Error ? error.message : String(error)
  const code = error instanceof JsonRpcGatewayError ? error.code : undefined

  return new ConnectorRpcError(message, code, reasonOf(error))
}

/** True when a failure is one of these reasons. The page branches on reasons,
 *  never on codes or message text. */
export function isConnectorReason(error: unknown, ...reasons: readonly ConnectorErrorReason[]): boolean {
  const reason = error instanceof ConnectorRpcError ? error.reason : reasonOf(error)

  return reason !== undefined && reasons.includes(reason)
}

const ACCOUNT_OWNER: AccountOwner = { type: 'account' }

// Both upstreams are called from the backend with a 30 s no-retry client. A 30 s
// renderer deadline would fire at the same instant and replace the typed reason
// with a bare timeout, so the page would show "failed" where the backend was
// about to say "signed out". Give the slow handler room to answer for itself.
const CONNECTOR_TIMEOUT_MS = 45_000

/** A capability scope split into the two arguments the gateway router takes. */
function routeOf(scope: ProfileScope): { connectionId: null | string; profile: string } {
  if (scope && typeof scope === 'object') {
    return {
      connectionId: (scope.connectionId ?? '').trim() || null,
      profile: (scope.profile ?? '').trim()
    }
  }

  return { connectionId: null, profile: (scope ?? '').trim() }
}

/** How a call may take the pool's reserved spawn slot. A read is ambient; a
 *  write is a person pressing a control and waiting for it. */
type Urgency = 'background' | 'foreground'

async function call<M extends keyof RpcMethods>(
  scope: ProfileScope,
  method: M,
  params: RpcMethods[M]['params'],
  urgency: Urgency = 'background'
): Promise<RpcMethods[M]['result']> {
  const { connectionId, profile } = routeOf(scope)

  try {
    return await requestGatewayForAgent<RpcMethods[M]['result']>(
      connectionId,
      profile,
      method,
      // The router owns `profile`: it injects the routed key, so the contract's
      // optional `profile` field is deliberately never set here.
      params as unknown as Record<string, unknown>,
      CONNECTOR_TIMEOUT_MS,
      undefined,
      { spawnPriority: urgency }
    )
  } catch (error) {
    throw asConnectorError(error)
  }
}

// ── reads ──────────────────────────────────────────────────────────────────

/** Connection state for every app the tool gateway knows about, plus the
 *  account-wide `available` flag that decides whether this page has a hosted
 *  half at all. */
export const listConnectors = (scope: ProfileScope) => call(scope, 'connectors.list', { owner: ACCOUNT_OWNER })

/** Names, descriptions and categories. The list RPC knows slugs; this knows words. */
export const connectorCatalog = (scope: ProfileScope) => call(scope, 'connectors.catalog', {})

/** The signed-in accounts behind the connections: label, status and when each started. */
export const connectorAccounts = (scope: ProfileScope, connector?: string) =>
  call(scope, 'connectors.accounts', connector === undefined ? {} : { connector })

/** Every visible policy layer, newest revision included. The member layer is the
 *  one this page writes; the rest are read to know what is locked. */
export const connectorPolicy = (scope: ProfileScope) => call(scope, 'connectors.policy.get', {})

/** One connector's tool list. The backend caches it for 24 h; `refresh` is the
 *  Refresh button asking for a revalidation. */
export const connectorTools = (scope: ProfileScope, slug: string, refresh = false) =>
  call(scope, 'connectors.tools', { refresh, slug })

/** The live snapshot of an account operation, for a window that joined one it
 *  did not start. */
export const accountOperationStatus = (scope: ProfileScope, opId: string) =>
  call(scope, 'connectors.operation.status', { op_id: opId, owner: ACCOUNT_OWNER })

// ── writes ─────────────────────────────────────────────────────────────────

/** Apply one member-layer change. `expectedRevision` is the compare-and-set
 *  token: the revision the person was looking at when they pressed Save. */
export const setConnectorPolicy = (
  scope: ProfileScope,
  change: ConnectorChange | ToolsChange,
  expectedRevision?: string
) =>
  call(
    scope,
    'connectors.policy.set',
    expectedRevision === undefined ? { change } : { change, expected_revision: expectedRevision },
    'foreground'
  )

/** Forget one signed-in account. The app stays in the catalog; only the
 *  connection goes. */
export const removeConnectorAccount = (scope: ProfileScope, connectionId: string) =>
  call(scope, 'connectors.accounts.remove', { connection_id: connectionId }, 'foreground')

/** Open (or re-mint) an authorization for one or more apps, owned by the
 *  account rather than by a chat. Answers the operation snapshot the connect
 *  element renders. */
export const connectAccountConnectors = (scope: ProfileScope, connectors: readonly string[], reconnect = false) =>
  call(scope, 'connectors.connect', { connectors: [...connectors], owner: ACCOUNT_OWNER, reconnect }, 'foreground')

/** The return from the browser: tell the backend to look at the operation now
 *  instead of waiting for its next poll. */
export const wakeAccountOperation = (scope: ProfileScope, opId: string) =>
  call(scope, 'connectors.operation.wake', { op_id: opId, owner: ACCOUNT_OWNER }, 'foreground')

/** The connect element's answer — a per-target outcome, or Continue for
 *  "Stop waiting". */
export const respondToAccountOperation = (scope: ProfileScope, opId: string, result: ConnectionAnswer) =>
  call(scope, 'connection.respond', { op_id: opId, owner: ACCOUNT_OWNER, result }, 'foreground')
