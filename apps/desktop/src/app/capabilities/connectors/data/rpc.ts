import type { RpcMethods } from '@rabbit/shared'
import { JsonRpcGatewayError } from '@rabbit/shared'

import type { ProfileScope } from '@/rabbit'
import { requestGatewayForAgent } from '@/store/gateway'

export class ConnectorRpcError extends Error {
  readonly code: number | undefined

  constructor(message: string, code?: number) {
    super(message)
    this.name = 'ConnectorRpcError'
    this.code = code
  }
}

const asError = (cause: unknown): Error => (cause instanceof Error ? cause : new Error(String(cause)))

export function asConnectorError(cause: unknown): ConnectorRpcError {
  const error = asError(cause)

  if (error instanceof ConnectorRpcError) {
    return error
  }

  const code = error instanceof JsonRpcGatewayError ? error.code : undefined

  return new ConnectorRpcError(error.message, code)
}

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

export const setMcpBearerToken = (scope: ProfileScope, name: string, value: string) =>
  call(scope, 'mcp.servers.set_api_key', { name, value }, 'foreground')

export const listMcpServers = (scope: ProfileScope) => call(scope, 'mcp.servers.list', {})

export const mcpServerStatus = (scope: ProfileScope) => call(scope, 'mcp.servers.status', {})
