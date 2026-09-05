import { resolveTestWsUrl } from './connection-config'
import { probeGatewayWebSocket } from './gateway-ws-probe'

const DEFAULT_STATUS_TIMEOUT_MS = 8_000
const DEFAULT_READY_TIMEOUT_MS = 45_000
const DEFAULT_RETRY_DELAY_MS = 500

interface RemoteConnection {
  baseUrl: string
  authMode?: string
  token?: string | null
  wsUrl?: string
}

interface StatusOptions {
  timeoutMs: number
}

interface RemoteReadinessDeps {
  fetchStatus: (baseUrl: string, token: string | null, options: StatusOptions) => Promise<unknown>
  mintTicket?: (baseUrl: string) => Promise<string>
  probeWebSocket?: typeof probeGatewayWebSocket
  WebSocketImpl?: any
  statusTimeoutMs?: number
  readyTimeoutMs?: number
  retryDelayMs?: number
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

function formatRemoteWebSocketFailure(authMode: 'token' | 'oauth', reason?: string) {
  const detail = reason ? ` ${reason}` : ''

  if (authMode === 'oauth') {
    return (
      'Remote Hermes backend is reachable, but the live /api/ws chat connection failed.' +
      detail +
      ' Sign in again from Settings → Gateway, or switch to Local.'
    )
  }

  return (
    'Remote Hermes backend is reachable, but the saved token could not open the live /api/ws chat connection.' +
    detail +
    ' Refresh the token or switch to Local.'
  )
}

async function checkRemoteHermesOnce<T extends RemoteConnection>(
  remote: T,
  deps: RemoteReadinessDeps
): Promise<T & { wsUrl: string }> {
  const authMode = remote.authMode === 'oauth' ? 'oauth' : 'token'
  const token = authMode === 'oauth' ? null : remote.token || null

  await deps.fetchStatus(remote.baseUrl, token, {
    timeoutMs: deps.statusTimeoutMs ?? DEFAULT_STATUS_TIMEOUT_MS
  })

  const probeUrl = await resolveTestWsUrl(remote.baseUrl, authMode, token, { mintTicket: deps.mintTicket })

  if (!probeUrl) {
    throw new Error(
      'Remote Hermes backend is reachable, but no saved session token is available for /api/ws. ' +
        'Refresh the token or switch to Local.'
    )
  }

  const probe = await (deps.probeWebSocket ?? probeGatewayWebSocket)(probeUrl, {
    WebSocketImpl: deps.WebSocketImpl ?? globalThis.WebSocket
  })

  if (!probe.ok) {
    throw new Error(formatRemoteWebSocketFailure(authMode, probe.reason))
  }

  // OAuth tickets are single-use. The readiness probe consumed its ticket, so
  // mint a separate one for the renderer instead of returning the probe URL.
  const wsUrl =
    authMode === 'oauth'
      ? await resolveTestWsUrl(remote.baseUrl, authMode, token, { mintTicket: deps.mintTicket })
      : probeUrl

  if (!wsUrl) {
    throw new Error('Remote gateway WebSocket credentials are unavailable.')
  }

  return { ...remote, wsUrl }
}

async function waitForRemoteHermes<T extends RemoteConnection>(
  remote: T,
  deps: RemoteReadinessDeps
): Promise<T & { wsUrl: string }> {
  if (!remote?.baseUrl) {
    throw new Error('Remote Hermes backend is not configured. Switch to Local or configure a remote gateway.')
  }

  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const deadline = now() + (deps.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS)
  let lastError: unknown = null

  do {
    try {
      return await checkRemoteHermesOnce(remote, deps)
    } catch (error) {
      lastError = error

      if (now() >= deadline) {
        break
      }

      await sleep(deps.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS)
    }
  } while (now() < deadline)

  const detail = lastError instanceof Error ? lastError.message : 'timeout'
  throw new Error(`Remote Hermes backend did not become ready: ${detail}`)
}

// Keep the two production callers explicit and independently testable. A
// profile-pool remote and the primary remote both have to pass the authenticated
// WebSocket leg before their respective boot paths may publish a connection.
function preparePooledRemoteBackend<T extends RemoteConnection>(remote: T, deps: RemoteReadinessDeps) {
  return waitForRemoteHermes(remote, deps)
}

function preparePrimaryRemoteBackend<T extends RemoteConnection>(remote: T, deps: RemoteReadinessDeps) {
  return waitForRemoteHermes(remote, deps)
}

export {
  checkRemoteHermesOnce,
  formatRemoteWebSocketFailure,
  preparePooledRemoteBackend,
  preparePrimaryRemoteBackend,
  waitForRemoteHermes
}

export type { RemoteConnection, RemoteReadinessDeps }
