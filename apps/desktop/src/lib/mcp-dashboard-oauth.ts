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
  sleep?: (milliseconds: number) => Promise<void>
  maxPollFailures?: number
}

const defaultSleep = (milliseconds: number) => new Promise<void>(resolve => window.setTimeout(resolve, milliseconds))

export async function completeMcpDesktopOAuth({
  serverName,
  start,
  status,
  openExternal,
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
