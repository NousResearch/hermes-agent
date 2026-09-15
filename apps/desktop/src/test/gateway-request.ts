import type { RpcMethods } from '@hermes/shared'
import { vi } from 'vitest'

export type RpcMethod = keyof RpcMethods

/** Per-method answers for a gateway double; each resolver sees the typed params of its method. */
export type RpcResolvers = Partial<{
  [M in RpcMethod]: (params: RpcMethods[M]['params']) => Promise<RpcMethods[M]['result']> | RpcMethods[M]['result']
}>

/** A `GatewayRequest` double that stays a vi.fn (call assertions work) and answers from `resolvers`. */
export function gatewayRequestMock(resolvers: RpcResolvers = {}) {
  return vi.fn(
    async <M extends RpcMethod>(method: M, params: RpcMethods[M]['params']): Promise<RpcMethods[M]['result']> => {
      const resolve = resolvers[method]

      if (!resolve) {
        throw new Error(`unexpected RPC ${method}`)
      }

      return resolve(params)
    }
  )
}
