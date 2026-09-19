import type { RpcMethods, ServerRequestMap } from './gateway-contract.generated.js'
import type { ServerRequest } from './json-rpc-channel.js'

declare function request<M extends keyof RpcMethods>(
  method: M,
  params: RpcMethods[M]['params']
): Promise<RpcMethods[M]['result']>
declare function onServerRequest<M extends keyof ServerRequestMap>(
  method: M,
  handler: (req: ServerRequest<M>) => void
): () => void

// @ts-expect-error typo in method name
void request('session.statsu', { session_id: 's' })
// @ts-expect-error unknown param
void request('session.status', { session_id: 's', extra: 1 })
// @ts-expect-error missing required param
void request('session.status', {})
request('session.status', { session_id: 's' }).then(r => {
  // @ts-expect-error wrong result field type
  const n: number = r.output

  return n
})
onServerRequest('clarify', req => {
  // @ts-expect-error un-narrowed access (questions only on the batch arm)
  void req.params.questions
  // @ts-expect-error wrong reply key
  req.respond({ value: 'x' })
})
