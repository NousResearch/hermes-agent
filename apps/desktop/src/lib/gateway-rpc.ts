import { JSON_RPC_METHOD_NOT_FOUND } from '@hermes/shared'

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

/** True when a JSON-RPC call failed because the backend rejects one of its
 *  params — e.g. `config.get` answering 4002 "unknown config key" for a key
 *  that predates this build's feature set. Distinguishable from a missing
 *  method (-32601): the method exists, the surface does not. */
export function isMissingRpcParamsKey(error: unknown): boolean {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  return typeof code === 'number' && code === 4002
}

export function isOutOfSyncRpcParams(error: Error | string): boolean {
  return /out of sync \(different versions\)/i.test(error.toString())
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

/** True when a projects.* RPC was refused because the profile's config turns
 *  the projects feature off (backend 5061 "projects disabled by config"). */
export function isProjectsDisabledByConfig(error: unknown): boolean {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined

  if (typeof code === 'number') {
    return code === 5061 && /disabled by config/i.test(error instanceof Error ? error.message : String(error))
  }

  return /projects disabled by config/i.test(error instanceof Error ? error.message : String(error ?? ''))
}

/** True when a pre-deferral backend refused a mid-turn model switch (4009).
 *  Current gateways park the pick and answer `scope: "pending"` instead. */
export function isBusySessionModelSwitch(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /session busy/i.test(message) && /switching models/i.test(message)
}
