import { capabilityScoped, type ProfileScope } from '@/api/client'
import { type McpOAuthFlow, mcpOAuthRpc } from '@/api/mcp'

import { browserBootstrap } from './browser-transport'
import { isMissingRpcMethod } from './gateway-rpc'
import { isBrowserHostedDesktop } from './platform'

interface CompleteOptions {
  serverName: string
  profile?: ProfileScope
  cancelled?: () => boolean
  sleep?: (milliseconds: number) => Promise<void>
  maxPollFailures?: number
  timeoutMs?: number
}

interface OAuthResult {
  ok: boolean
  session_id?: string
  auth_url?: string
  status?: 'pending' | 'approved' | 'error'
  error_message?: string
  tools?: McpOAuthFlow['tools']
}

/** Deliberate cancellation is not an error toast. */
export class McpOAuthCancelled extends Error {
  constructor() {
    super('OAuth cancelled by user')
    this.name = 'McpOAuthCancelled'
  }
}

const defaultSleep = (milliseconds: number) => new Promise<void>(resolve => window.setTimeout(resolve, milliseconds))
const UPDATE_BACKEND = 'Update the Hermes backend to support Desktop MCP OAuth callbacks.'

function isGatewayLoopbackCallback(redirect: URL): boolean {
  return redirect.protocol === 'http:' && redirect.hostname === '127.0.0.1' &&
    Boolean(redirect.port) && Number(redirect.port) !== 0 && redirect.pathname === '/callback' &&
    !redirect.username && !redirect.password && !redirect.search && !redirect.hash
}

/** This Webapp's own callback: its origin and base path, then the server's name. */
function isWebappPublicCallback(redirect: URL, serverName: string): boolean {
  const prefix = new URL(`${browserBootstrap()?.basePath || ''}/api/mcp/oauth/callback/`, window.location.origin).pathname

  if (redirect.origin !== window.location.origin || !redirect.pathname.startsWith(prefix) ||
    redirect.username || redirect.password || redirect.search || redirect.hash) {
    return false
  }

  try {
    return decodeURIComponent(redirect.pathname.slice(prefix.length)) === serverName
  } catch {
    return false
  }
}

/** Native clients relay through their bridge; remote Webapp tabs use their public callback. */
export async function completeMcpDesktopOAuth({
  serverName,
  profile,
  cancelled,
  sleep = defaultSleep,
  maxPollFailures = 3,
  timeoutMs = 360_000
}: CompleteOptions): Promise<McpOAuthFlow> {
  const deadline = Date.now() + timeoutMs
  const scope = capabilityScoped(profile)
  const rpc = mcpOAuthRpc(scope)
  const bridge = window.hermesDesktop.mcpOauth

  const publicCallback = !bridge && isBrowserHostedDesktop() &&
    !['127.0.0.1', 'localhost', '::1', '[::1]'].includes(window.location.hostname)

  // A legacy null connection can resolve to a remote registry primary.
  if (!bridge && scope.connectionId !== 'local') {
    throw new Error('Update Hermes Desktop to support MCP OAuth callbacks.')
  }

  let listener: { id: string; redirectUri: string } | undefined
  let sessionId: string | undefined
  let authUrl: string | undefined
  let approved = false
  let closed = false
  let relayError: unknown

  const checkCancelled = () => {
    if (cancelled?.()) {
      throw new McpOAuthCancelled()
    }
  }

  const request = async (action: 'start' | 'poll' | 'callback' | 'cancel', params: Record<string, unknown>) => {
    const result = await rpc<OAuthResult>(action, { name: serverName, ...params })

    if (!result.ok) {
      throw new Error(result.error_message || 'MCP OAuth request failed')
    }

    return result
  }

  try {
    checkCancelled()

    if (bridge) {
      listener = await bridge.listen()
      checkCancelled()
    }

    // The gateway uses its public callback only when this page is served from its public URL.
    const pageBase = `${window.location.origin}${browserBootstrap()?.basePath || ''}`
    const callbackParams = publicCallback ? { public_callback: pageBase } : {}
    const started = await request('start', listener ? { client_redirect_uri: listener.redirectUri } : callbackParams)
    sessionId = started.session_id
    authUrl = started.auth_url
    // Start may have created a flow while the user cancelled. Keep its id so
    // finally can release it, but do not launch a browser after cancellation.
    checkCancelled()

    if (!sessionId || !authUrl) {
      throw new Error('OAuth server did not provide an authorization URL and session')
    }

    // Pre-relay backends silently ignore unknown start params. Do not open an
    // authorization URL whose DCR/PKCE flow points at a different machine.
    const redirectUri = new URL(authUrl).searchParams.get('redirect_uri')

    if (listener && redirectUri !== listener.redirectUri) {
      throw new Error(UPDATE_BACKEND)
    }

    if (!listener) {
      // Match the existing backend-hosted listener, not an arbitrary local URL. A gateway
      // without webapp.public_url keeps it for a public-callback request too.
      const redirect = new URL(redirectUri || '')

      if (!isGatewayLoopbackCallback(redirect) && !(publicCallback && isWebappPublicCallback(redirect, serverName))) {
        throw new Error(`OAuth server did not provide a local loopback${publicCallback ? ' or Webapp' : ''} callback URL`)
      }
    }

    const flowId = sessionId

    if (bridge && listener) {
      void bridge
        .wait(listener.id)
        .then(async callback => {
          if (closed || cancelled?.()) {
            return
          }

          if (!callback.state) {
            throw new Error(callback.error || 'OAuth callback did not include state')
          }

          // Omit a null iss: a backend that predates the RFC 9207 relay rejects unknown params (4000).
          const { iss, ...rest } = callback
          await request('callback', { session_id: flowId, ...rest, ...(iss ? { iss } : {}) })
        })
        .catch(error => {
          relayError = error
        })
    }

    await window.hermesDesktop.openExternal(authUrl)
    let pollFailures = 0

    for (;;) {
      checkCancelled()

      if (Date.now() >= deadline) {
        throw new Error('Timed out waiting for MCP OAuth authorization')
      }

      if (relayError) {
        throw relayError
      }

      let current: OAuthResult

      try {
        current = await request('poll', { session_id: flowId })
        pollFailures = 0
      } catch (error) {
        if (++pollFailures >= maxPollFailures) {
          throw error
        }

        await sleep(1000)

        continue
      }

      checkCancelled()

      if (relayError) {
        throw relayError
      }

      if (current.status === 'approved') {
        approved = true

        return {
          flow_id: flowId,
          server_name: serverName,
          status: 'approved',
          authorization_url: authUrl,
          error: null,
          tools: current.tools
        }
      }

      if (current.status === 'error') {
        throw new Error(current.error_message || 'OAuth authorization failed')
      }

      await sleep(1000)
    }
  } catch (error) {
    if (isMissingRpcMethod(error)) {
      throw new Error(UPDATE_BACKEND)
    }

    throw error
  } finally {
    closed = true

    // Stop the native waiter first: its synthetic cancellation is NOT a
    // provider callback and must never race cleanup back onto the gateway.
    if (bridge && listener) {
      await bridge.cancel(listener.id).catch(() => {})
    }

    if (sessionId && !approved) {
      try {
        await request('cancel', { session_id: sessionId })
      } catch (error) {
        // Relay-capable backends predating oauth.cancel can still release a
        // waiting worker with a state-checked denial. No HTTP redirect retry.
        if (isMissingRpcMethod(error) && authUrl) {
          const state = new URL(authUrl).searchParams.get('state')

          if (state) {
            await request('callback', { session_id: sessionId, state, error: 'access_denied' }).catch(() => {})
          }
        }
        // Network cleanup is best-effort; backend callback timeout is bounded.
      }
    }
  }
}
