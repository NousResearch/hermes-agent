import { JsonRpcGatewayError } from '@hermes/shared'

/** True when a JSON-RPC call failed because the backend predates the method. */
export function isMissingRpcMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

/** User-facing text for a gateway RPC failure. Prefer the server message; keep
 *  the numeric code only as a diagnostic suffix when the message is missing or
 *  already the opaque `Hermes RPC request failed (N)` fallback. */
export function formatGatewayRpcError(error: unknown, fallback = 'request failed'): string {
  if (error instanceof JsonRpcGatewayError) {
    const message = typeof error.message === 'string' ? error.message.trim() : ''

    if (message) {
      return message
    }

    if (typeof error.code === 'number') {
      return `Hermes RPC request failed (${error.code})`
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message.trim()
  }

  if (typeof error === 'string' && error.trim()) {
    return error.trim()
  }

  return fallback
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
