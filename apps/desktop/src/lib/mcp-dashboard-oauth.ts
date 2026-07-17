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
}

const defaultSleep = (milliseconds: number) =>
  new Promise<void>(resolve => window.setTimeout(resolve, milliseconds))

export async function completeMcpDesktopOAuth({
  serverName,
  start,
  status,
  openExternal,
  sleep = defaultSleep
}: CompleteOptions): Promise<McpOAuthFlow> {
  const started = await start(serverName)

  if (started.status === 'error') {
    throw new Error(started.error || 'OAuth failed to start')
  }

  if (!started.authorization_url) {
    throw new Error('OAuth server did not provide an authorization URL')
  }

  await openExternal(started.authorization_url)

  for (;;) {
    const current = await status(started.flow_id)

    if (current.status === 'approved') {
      return current
    }

    if (current.status === 'error') {
      throw new Error(current.error || 'OAuth authorization failed')
    }

    await sleep(1000)
  }
}