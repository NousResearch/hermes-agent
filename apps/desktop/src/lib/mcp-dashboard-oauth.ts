export interface McpOAuthFlow {
  flow_id: string
  server_name: string
  status: 'starting' | 'authorization_required' | 'approved' | 'error'
  authorization_url: string | null
  error: string | null
  tools?: Array<{ name: string; description: string }>
}

interface CompleteOptions {
  serverName: string
  start: (name: string) => Promise<McpOAuthFlow>
  status: (flowId: string) => Promise<McpOAuthFlow>
  openExternal: (url: string) => Promise<void>
  /** Polled between status checks. Returning true cancels: the flow is
   *  cancelled SERVER-SIDE (freeing the per-server in-progress slot — without
   *  it a retry 409s until the backend's callback timeout) and the promise
   *  rejects with `McpOAuthCancelled`. */
  cancelled?: () => boolean
  /** Server-side flow cancel, wired to DELETE /api/mcp/oauth/flows/{id}. */
  cancel?: (flowId: string) => Promise<unknown>
  sleep?: (milliseconds: number) => Promise<void>
  maxPollFailures?: number
}

/** Thrown when the caller's `cancelled()` tripped — callers branch on this to
 *  skip error toasts for a deliberate user cancel. */
export class McpOAuthCancelled extends Error {
  constructor() {
    super('OAuth cancelled by user')
    this.name = 'McpOAuthCancelled'
  }
}

const defaultSleep = (milliseconds: number) => new Promise<void>(resolve => window.setTimeout(resolve, milliseconds))

export async function completeMcpDesktopOAuth({
  serverName,
  start,
  status,
  openExternal,
  cancelled,
  cancel,
  sleep = defaultSleep,
  maxPollFailures = 3
}: CompleteOptions): Promise<McpOAuthFlow> {
  const started = await start(serverName)

  if (started.status === 'error') {
    throw new Error(started.error || 'OAuth failed to start')
  }

  if (!started.authorization_url) {
    throw new Error('OAuth server did not provide an authorization URL')
  }

  await openExternal(started.authorization_url)

  let pollFailures = 0

  for (;;) {
    if (cancelled?.()) {
      // Free the backend slot before rejecting; best-effort — the flow also
      // dies on its own timeout if this request is lost.
      await cancel?.(started.flow_id).catch(() => {})
      throw new McpOAuthCancelled()
    }

    let current: McpOAuthFlow

    try {
      current = await status(started.flow_id)
      pollFailures = 0
    } catch (error) {
      pollFailures += 1

      if (pollFailures >= maxPollFailures) {
        throw error
      }

      await sleep(1000)

      continue
    }

    if (current.status === 'approved') {
      return current
    }

    if (current.status === 'error') {
      throw new Error(current.error || 'OAuth authorization failed')
    }

    await sleep(1000)
  }
}
