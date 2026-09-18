import type { RpcMethod, RpcMethods } from '@hermes/shared'

import { requestGatewayForAgent } from '@/store/gateway'
import type { SessionOwnerRoute } from '@/store/session-request-router'

/** The `session.foreign.*` RPCs the import dialog drives, derived from the generated contract. */
export type ForeignMethod = Extract<RpcMethod, `session.foreign.${string}`>

export function foreignRequest<M extends ForeignMethod>(
  owner: SessionOwnerRoute,
  method: M,
  params: RpcMethods[M]['params'],
  signal?: AbortSignal
): Promise<RpcMethods[M]['result']> {
  return requestGatewayForAgent(owner.connectionId, owner.profile, method, params, 60_000, signal)
}
