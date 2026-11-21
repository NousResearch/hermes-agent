import { capabilityScoped } from '@/api/client'
import { type McpOAuthFlow } from '@/api/mcp'

import { completeMcpDesktopOAuth, type CompleteOptions, McpOAuthCancelled } from './mcp-dashboard-oauth'
import { isBrowserHostedDesktop } from './platform'

const defaultSleep = (milliseconds: number) => new Promise<void>(resolve => window.setTimeout(resolve, milliseconds))

/** MCP OAuth for whichever host this page runs in.
 *
 *  The browser-hosted Desktop (Webapp) usually runs on a different machine from the gateway and
 *  has no native callback bridge, so a loopback redirect (gateway-hosted or client-hosted) can
 *  never reach it. There the gateway's REST flow is used: it redirects the provider to its own
 *  public callback route. The native app keeps its existing flow. */
export function completeMcpOAuth(options: CompleteOptions): Promise<McpOAuthFlow> {
  return isBrowserHostedDesktop() ? completeMcpBrowserOAuth(options) : completeMcpDesktopOAuth(options)
}

type DesktopApi = Window['hermesDesktop']['api']

interface BrowserFlow {
  api: DesktopApi
  flowId: string
  profile: ReturnType<typeof capabilityScoped>['profile']
}

interface PollLimits {
  checkCancelled: () => void
  deadline: number
  maxPollFailures: number
  sleep: (milliseconds: number) => Promise<void>
}

const flowPath = (flowId: string) => `/api/mcp/oauth/flows/${encodeURIComponent(flowId)}`
const scopedTo = (profile: BrowserFlow['profile']) => (profile ? { profile } : {})

/** Start the gateway-hosted flow. */
function startBrowserFlow(api: DesktopApi, serverName: string, profile: BrowserFlow['profile']) {
  return api<McpOAuthFlow>({
    path: `/api/mcp/servers/${encodeURIComponent(serverName)}/auth`,
    method: 'POST',
    ...scopedTo(profile),
    timeoutMs: 60_000
  })
}

function authorizationOf(started: McpOAuthFlow): { authUrl: string; flowId: string } {
  if (started.status === 'error') {
    throw new Error(started.error || 'MCP OAuth failed to start')
  }

  if (!started.flow_id || !started.authorization_url) {
    throw new Error('OAuth server did not provide an authorization URL')
  }

  return { authUrl: started.authorization_url, flowId: started.flow_id }
}

/** Poll until the gateway reports the flow approved; throws on a provider error, cancel or timeout. */
async function awaitBrowserFlow({ api, flowId, profile }: BrowserFlow, limits: PollLimits): Promise<McpOAuthFlow> {
  let pollFailures = 0

  for (;;) {
    limits.checkCancelled()

    if (Date.now() >= limits.deadline) {
      throw new Error('Timed out waiting for MCP OAuth authorization')
    }

    let current: McpOAuthFlow | undefined

    try {
      current = await api<McpOAuthFlow>({ path: flowPath(flowId), ...scopedTo(profile) })
      pollFailures = 0
    } catch (error) {
      if (++pollFailures >= limits.maxPollFailures) {
        throw error
      }
    }

    if (current?.status === 'approved') {
      return current
    }

    if (current?.status === 'error') {
      throw new Error(current.error || 'OAuth authorization failed')
    }

    await limits.sleep(1000)
  }
}

/** Browser-hosted Desktop: the gateway runs the flow and receives the provider's redirect on its
 *  public callback route; this tab only opens the authorization page and polls for the outcome. */
async function completeMcpBrowserOAuth({
  serverName,
  profile: requestedProfile,
  cancelled,
  sleep = defaultSleep,
  maxPollFailures = 3,
  timeoutMs = 360_000
}: CompleteOptions): Promise<McpOAuthFlow> {
  const deadline = Date.now() + timeoutMs
  const api = window.hermesDesktop.api
  const { profile } = capabilityScoped(requestedProfile)
  // Opened now, while the click's user activation lasts; the URL arrives after an async start.
  const popup = window.open('', '_blank')
  let flowId: string | undefined
  let approved = false

  const checkCancelled = () => {
    if (cancelled?.()) {
      throw new McpOAuthCancelled()
    }
  }

  try {
    checkCancelled()

    const started = await startBrowserFlow(api, serverName, profile)

    flowId = started.flow_id
    checkCancelled()

    const { authUrl, flowId: startedFlowId } = authorizationOf(started)

    if (popup && !popup.closed) {
      popup.opener = null
      popup.location.href = authUrl
    } else {
      await window.hermesDesktop.openExternal(authUrl)
    }

    const current = await awaitBrowserFlow(
      { api, flowId: startedFlowId, profile },
      { checkCancelled, deadline, maxPollFailures, sleep }
    )

    approved = true

    return { ...current, server_name: serverName, authorization_url: authUrl, error: null }
  } finally {
    if (!approved) {
      if (popup && !popup.closed) {
        popup.close()
      }

      if (flowId) {
        await api({ path: flowPath(flowId), method: 'DELETE', ...scopedTo(profile) }).catch(() => {})
      }
    }
  }
}
