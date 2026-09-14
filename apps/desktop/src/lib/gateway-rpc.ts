import { JSON_RPC_METHOD_NOT_FOUND, type RpcMethods } from '@hermes/shared'

/** True when a JSON-RPC call failed because the backend predates the method.
 *  The gateway answers -32601 (`tui_gateway/server.py::dispatch`) and the
 *  shared client keeps that code on the error; the message match is only for
 *  errors that lost their frame across the IPC bridge or a wrapped rethrow. */
export function isMissingRpcMethod(error: unknown): boolean {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  if (typeof code === 'number') {
    return code === JSON_RPC_METHOD_NOT_FOUND
  }

  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

/** REST twin of isMissingRpcMethod: the route does not exist on this backend.
 *  Matches the backend catch-all ('404: {"detail":"No such API endpoint: …}'),
 *  FastAPI's bare 404 on headless serve — directly, or wrapped as "Error
 *  invoking remote method 'hermes:api': Error: 404: …" through the IPC bridge
 *  — and the Electron JSON-guard ("endpoint is likely missing"). Transient
 *  failures (timeouts, 5xx, connection refused) must NOT match: they are
 *  retryable, not a capability verdict. Only sound for calls where a 404 can
 *  mean nothing else — a route with path params can 404 on a bad id. */
export function isMissingRestEndpoint(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return (
    /no such api endpoint/i.test(message) ||
    /endpoint is likely missing/i.test(message) ||
    /(?:^\s*|error:\s*)404\b/i.test(message)
  )
}

/** True when a prompt response raced a backend-side timeout / completion. */
export function isMissingPendingPromptRequest(error: unknown, key: string): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return message.toLowerCase().includes(`no pending ${key.toLowerCase()} request`)
}

/** True when a pre-deferral backend refused a mid-turn model switch (4009).
 *  Current gateways park the pick and answer `scope: "pending"` instead. */
export function isBusySessionModelSwitch(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /session busy/i.test(message) && /switching models/i.test(message)
}

/** The typed RPC caller every desktop request wrapper forwards; the method literal picks params and result. */
export type GatewayRequest = <M extends keyof RpcMethods>(
  method: M,
  params: RpcMethods[M]['params'],
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<RpcMethods[M]['result']>

/** The routing keys the generated params carry: `profile` on every method, `session_id` on the session-scoped ones. */
export interface RoutableParams {
  profile?: null | string
  session_id?: null | string
}

/** The session a session-scoped RPC names, read off the generated params. */
export function paramsSessionId(params: RoutableParams): string {
  return (params.session_id ?? '').trim()
}
