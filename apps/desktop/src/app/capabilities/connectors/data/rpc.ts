import type {
  AccountOwner,
  ConnectionAnswer,
  ConnectorChange,
  ConnectorErrorReason,
  RpcMethods,
  ToolsChange
} from '@hermes/shared'
import { JsonRpcGatewayError } from '@hermes/shared'

import type { ProfileScope } from '@/hermes'
import { requestGatewayForAgent } from '@/store/gateway'

export type { ConnectorErrorReason }

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

interface ConnectorErrorData {
  reason?: ConnectorErrorReason
}

function reasonOf(error: Error): ConnectorErrorReason | undefined {
  if (!(error instanceof JsonRpcGatewayError)) {
    return undefined
  }

  // SAFETY: `error.data` is the gateway's own `error.data` object; an absent or malformed `reason` reads as undefined.
  const data = (error.data ?? {}) as ConnectorErrorData

  return data.reason ?? undefined
}

const asError = (cause: unknown): Error => (cause instanceof Error ? cause : new Error(String(cause)))

export function asConnectorError(cause: unknown): ConnectorRpcError {
  const error = asError(cause)

  if (error instanceof ConnectorRpcError) {
    return error
  }

  const code = error instanceof JsonRpcGatewayError ? error.code : undefined

  return new ConnectorRpcError(error.message, code, reasonOf(error))
}

export function isConnectorReason(cause: unknown, ...reasons: readonly ConnectorErrorReason[]): boolean {
  const error = asError(cause)
  const reason = error instanceof ConnectorRpcError ? error.reason : reasonOf(error)

  return reason !== undefined && reasons.includes(reason)
}

const ACCOUNT_OWNER: AccountOwner = { type: 'account' }

const CONNECTOR_TIMEOUT_MS = 45_000

interface GatewayRoute {
  connectionId: null | string
  profile: string
}

function routeOf(scope: ProfileScope): GatewayRoute {
  if (scope instanceof Object) {
    return {
      connectionId: (scope.connectionId ?? '').trim() || null,
      profile: (scope.profile ?? '').trim()
    }
  }

  return { connectionId: null, profile: (scope ?? '').trim() }
}

type Urgency = 'background' | 'foreground'

type GatewayParams = NonNullable<Parameters<typeof requestGatewayForAgent>[3]>

async function call<M extends keyof RpcMethods>(
  scope: ProfileScope,
  method: M,
  params: RpcMethods[M]['params'],
  urgency: Urgency = 'background'
): Promise<RpcMethods[M]['result']> {
  const { connectionId, profile } = routeOf(scope)

  try {
    // SAFETY: every `RpcMethods[M]['params']` is a generated object type, which is the record the router takes.
    const payload = params as RpcMethods[M]['params'] & GatewayParams

    return await requestGatewayForAgent<RpcMethods[M]['result']>(
      connectionId,
      profile,
      method,
      payload,
      CONNECTOR_TIMEOUT_MS,
      undefined,
      { spawnPriority: urgency }
    )
  } catch (error) {
    throw asConnectorError(error)
  }
}

export const listConnectors = (scope: ProfileScope) => call(scope, 'connectors.list', { owner: ACCOUNT_OWNER })

export const connectorCatalog = (scope: ProfileScope) => call(scope, 'connectors.catalog', {})

export const connectorAccounts = (scope: ProfileScope, connector?: string) =>
  call(scope, 'connectors.accounts', connector === undefined ? {} : { connector })

export const connectorPolicy = (scope: ProfileScope) => call(scope, 'connectors.policy.get', {})

export const connectorTools = (scope: ProfileScope, slug: string, refresh = false) =>
  call(scope, 'connectors.tools', { refresh, slug })

export const accountOperationStatus = (scope: ProfileScope, opId: string) =>
  call(scope, 'connectors.operation.status', { op_id: opId, owner: ACCOUNT_OWNER })

export const setConnectorPolicy = (
  scope: ProfileScope,
  change: ConnectorChange | ToolsChange,
  expectedRevision: string
) => call(scope, 'connectors.policy.set', { change, expected_revision: expectedRevision }, 'foreground')

export const removeConnectorAccount = (scope: ProfileScope, connectionId: string) =>
  call(scope, 'connectors.accounts.remove', { connection_id: connectionId }, 'foreground')

export const connectAccountConnectors = (scope: ProfileScope, connectors: readonly string[], reconnect = false) =>
  call(scope, 'connectors.connect', { connectors: [...connectors], owner: ACCOUNT_OWNER, reconnect }, 'foreground')

export const wakeAccountOperation = (scope: ProfileScope, opId: string) =>
  call(scope, 'connectors.operation.wake', { op_id: opId, owner: ACCOUNT_OWNER }, 'foreground')

export const respondToAccountOperation = (scope: ProfileScope, opId: string, result: ConnectionAnswer) =>
  call(scope, 'connection.respond', { op_id: opId, owner: ACCOUNT_OWNER, result }, 'foreground')

export const setMcpBearerToken = (scope: ProfileScope, name: string, value: string) =>
  call(scope, 'mcp.servers.set_api_key', { name, value }, 'foreground')

export const listMcpServers = (scope: ProfileScope) => call(scope, 'mcp.servers.list', {})

export const mcpServerStatus = (scope: ProfileScope) => call(scope, 'mcp.servers.status', {})
