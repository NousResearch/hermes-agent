import { capabilityScoped, type ProfileScope } from '../src/api/client'
import type { McpOAuthFlow } from '../src/api/mcp'

import { browserMessage } from './messages'
import { connection, requestJson } from './transport'

interface CompleteOptions {
  serverName: string
  profile?: ProfileScope
  cancelled?: () => boolean
  sleep?: (milliseconds: number) => Promise<void>
  maxPollFailures?: number
  timeoutMs?: number
}

export class McpOAuthCancelled extends Error {
  constructor() {
    super(browserMessage('oauthCancelled'))
    this.name = 'McpOAuthCancelled'
  }
}

/** Own the popup separately from the backend's OAuth flow lifetime. */
function reserveAuthorizationWindow() {
  // Reserve while the original click still grants user activation.
  const popup = window.open('about:blank', '_blank')

  if (popup) {popup.opener = null}

  return {
    handedOff: false,
    navigate(value: string) {
      const url = new URL(value)

      if (!['http:', 'https:'].includes(url.protocol)) {throw new Error(browserMessage('httpLinks'))}

      if (popup && !popup.closed) {
        popup.location.replace(url.toString())
      } else {
        // The backend callback completes independently after same-tab navigation.
        window.location.assign(url.toString())
        this.handedOff = true
      }
    },
    close() { if (popup && !popup.closed) {popup.close()} }
  }
}

/** Use the dashboard's public callback, never a listener on the phone or VPS loopback. */
export async function completeMcpDesktopOAuth({
  serverName, profile, cancelled,
  sleep = milliseconds => new Promise(resolve => window.setTimeout(resolve, milliseconds)),
  maxPollFailures = 3, timeoutMs = 360_000
}: CompleteOptions): Promise<McpOAuthFlow> {
  const scope = capabilityScoped(profile)
  const owner = { ...scope, profile: scope.profile || connection().profile || 'default' }
  const deadline = Date.now() + timeoutMs
  let flowId: string | undefined
  let approved = false
  let opened = false
  let authorization: ReturnType<typeof reserveAuthorizationWindow> | undefined

  const checkCancelled = () => { if (cancelled?.()) {throw new McpOAuthCancelled()} }

  try {
    checkCancelled()
    authorization = reserveAuthorizationWindow()
    let flow = await requestJson<McpOAuthFlow>({ ...owner, path: `/api/mcp/servers/${encodeURIComponent(serverName)}/auth`, method: 'POST' })
    flowId = flow.flow_id

    if (!flowId) {throw new Error(browserMessage('oauthMissingFlow'))}
    let failures = 0

    for (;;) {
      checkCancelled()

      if (Date.now() >= deadline) {throw new Error(browserMessage('oauthTimeout'))}

      if (flow.status === 'approved') { approved = true;

 return flow }

      if (flow.status === 'error') {throw new Error(flow.error || browserMessage('oauthFailed'))}

      if (!opened && flow.authorization_url) {
        authorization.navigate(flow.authorization_url)
        opened = true
      }

      await sleep(1000)
      checkCancelled()

      try {
        flow = await requestJson<McpOAuthFlow>({ ...owner, path: `/api/mcp/oauth/flows/${encodeURIComponent(flowId)}` })
        failures = 0
      } catch (error) {
        if (++failures >= maxPollFailures) {throw error}
      }
    }
  } finally {
    authorization?.close()

    if (flowId && !approved && !authorization?.handedOff) {
      await requestJson({ ...owner, path: `/api/mcp/oauth/flows/${encodeURIComponent(flowId)}`, method: 'DELETE' }).catch(() => {})
    }
  }
}
