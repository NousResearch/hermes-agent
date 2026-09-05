/**
 * Desktop's managed Computer Use bridge.
 *
 * When the app drives a remote backend, `computer_use` runs where the agent
 * runs — the VPS — while the person watching is here. This closes that gap
 * without asking anyone to tunnel anything: the app starts a loopback-only
 * `hermes computer-use bridge` sidecar beside itself, opens an authenticated
 * WebSocket *outward* to the backend, and answers the backend's clicks over
 * that socket. The backend never dials the laptop, so NAT and firewalls are
 * not in the way.
 *
 * Two properties are load-bearing:
 *
 * - **Scoped, never global.** Each remote+profile pair owns its own socket and
 *   its own ownership record. The backend files a bridge under the identity
 *   its credential carried and only matches a session with the same identity,
 *   so a socket opened for one profile must not be reused as another's.
 * - **A lite client says so.** A Desktop install with no local Hermes runtime
 *   cannot run the sidecar. That is a normal state, not an error to retry
 *   forever: the bridge reports it once and stays off, and the backend simply
 *   never sees a bridge and keeps driving its own machine.
 */

import crypto from 'crypto'
import http from 'http'
import https from 'https'

import {
  releaseBridgeOwnerAndStopSidecarIfIdle,
  scheduleBridgeReconnectIfCurrent,
  ScopedComputerUseBridgeLifecycle
} from './computer-use-bridge-lifecycle'
import {
  buildComputerUseBridgeWsUrl,
  buildComputerUseBridgeWsUrlWithTicket,
  connectionScopeKey
} from './connection-config'

const LOCAL_TIMEOUT_MS = 30_000
const START_TIMEOUT_MS = 20_000
const CONNECT_TIMEOUT_MS = 8_000
const RECONNECT_MS = 5_000
const TOKEN_ENV = 'HERMES_COMPUTER_USE_BRIDGE_TOKEN'

const NO_LOCAL_RUNTIME =
  'This Hermes Desktop has no local agent runtime, so it cannot run the Computer Use bridge. ' +
  'computer_use will keep driving the backend machine.'

/** The one owner key for the app's own connection, as opposed to a pooled profile. */
const PRIMARY_OWNER = 'primary'

interface RemoteTarget {
  baseUrl: string
  authMode?: string
  token?: string
  source?: string
  computerUseBridge?: boolean
}

interface BridgeConnection {
  url: string
  token: string
  remoteKey: string
  remoteBaseUrl: string
  profile: string | null
  ws: any
  closedByDesktop?: boolean
}

export interface ComputerUseBridgeDeps {
  /** Resolve the `hermes` invocation for the given argv, as the backend spawn does. */
  resolveHermesBackend: (args: string[]) => any
  /** Materialize a managed runtime if one is pending. */
  ensureRuntime: (backend: any) => Promise<any>
  resolveHermesCwd: () => string
  hiddenWindowsChildOptions: (options: any) => any
  stopBackendChild: (child: any) => void
  rememberLog: (line: string) => void
  mintGatewayWsTicket: (baseUrl: string) => Promise<string>
  spawn: (command: string, args: string[], options: any) => any
  hermesHome: string
  webSocketImpl?: any
}

export function createComputerUseBridge(deps: ComputerUseBridgeDeps) {
  const lifecycle = new ScopedComputerUseBridgeLifecycle<BridgeConnection>()

  let sidecar: any = null
  let sidecarStart: Promise<{ url: string; token: string }> | null = null
  let sidecarInfo: { url: string; token: string } | null = null
  let stopping = false
  let generation = 0
  let primaryRemoteKey: string | null = null
  // A lite install has no runtime to spawn. Latch it so a reconnect loop does
  // not re-derive the same answer every five seconds for the life of the app.
  let unsupportedReason: string | null = null

  const log = (line: string) => deps.rememberLog(`[computer-use-bridge] ${line}`)

  function remoteKeyFor(remote: RemoteTarget, profile: string | null): string {
    const scoped = remote.source === 'profile' ? null : profile

    return `${remote.baseUrl}|${remote.authMode || ''}|${remote.source || ''}|${connectionScopeKey(scoped) || 'current'}`
  }

  function request(baseUrl: string, token: string, pathSuffix: string, options: any = {}): Promise<any> {
    return new Promise((resolve, reject) => {
      const body = options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body))
      const parsed = new URL(`${baseUrl}${pathSuffix}`)
      const client = parsed.protocol === 'https:' ? https : http

      const req = client.request(
        parsed,
        {
          method: options.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
            ...(body ? { 'Content-Length': String(body.length) } : {})
          }
        },
        res => {
          const chunks: Buffer[] = []
          res.on('error', reject)
          res.on('data', chunk => chunks.push(chunk))
          res.on('end', () => {
            const text = Buffer.concat(chunks).toString('utf8')

            if ((res.statusCode || 500) >= 400) {
              reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`))

              return
            }

            try {
              resolve(text ? JSON.parse(text) : null)
            } catch {
              reject(new Error(`Invalid JSON from local Computer Use bridge: ${text.slice(0, 200)}`))
            }
          })
        }
      )

      req.on('error', reject)
      req.setTimeout(LOCAL_TIMEOUT_MS, () => {
        req.destroy(new Error(`Timed out waiting for local Computer Use bridge after ${LOCAL_TIMEOUT_MS}ms`))
      })

      if (body) {
        req.write(body)
      }

      req.end()
    })
  }

  function urlFromStdout(buffered: string): string | null {
    const match = String(buffered || '').match(/Hermes Computer Use bridge listening on (http:\/\/\S+)/)

    return match ? match[1].replace(/\/+$/, '') : null
  }

  function stopSidecar(): void {
    deps.stopBackendChild(sidecar)
    sidecar = null
    sidecarStart = null
    sidecarInfo = null
  }

  /**
   * Why this install cannot run the bridge at all, or null while it can.
   *
   * Cheap and side-effect free — it resolves the same argv the sidecar would
   * spawn and looks at whether a runtime is there — so the settings panel can
   * tell a lite client the truth before anyone toggles anything.
   */
  function unsupported(): string | null {
    if (unsupportedReason) {
      return unsupportedReason
    }

    try {
      if (deps.resolveHermesBackend(['computer-use', 'bridge']).bootstrap) {
        unsupportedReason = NO_LOCAL_RUNTIME
      }
    } catch (error: any) {
      unsupportedReason = `Computer Use bridge is unavailable here: ${error.message || error}`
    }

    return unsupportedReason
  }

  async function ensureSidecar(): Promise<{ url: string; token: string }> {
    if (unsupportedReason) {
      throw new Error(unsupportedReason)
    }

    if (sidecarInfo && sidecar && !sidecar.killed) {
      return sidecarInfo
    }

    if (sidecarStart) {
      return sidecarStart
    }

    sidecarStart = (async () => {
      const token = crypto.randomBytes(32).toString('base64url')
      const args = ['computer-use', 'bridge', '--host', '127.0.0.1', '--port', '0', '--token-env', TOKEN_ENV]
      let backend

      try {
        backend = deps.resolveHermesBackend(args)

        if (backend.bootstrap) {
          // Nothing to install our way out of here: the app is deliberately
          // running without a local agent runtime.
          unsupportedReason = NO_LOCAL_RUNTIME
          throw new Error(unsupportedReason)
        }

        backend = await deps.ensureRuntime(backend)
      } catch (error: any) {
        sidecarStart = null
        throw error
      }

      return await new Promise<{ url: string; token: string }>((resolve, reject) => {
        const cwd = deps.resolveHermesCwd()

        const child = deps.spawn(
          backend.command,
          backend.args,
          deps.hiddenWindowsChildOptions({
            cwd,
            env: {
              ...process.env,
              HERMES_HOME: deps.hermesHome,
              ...backend.env,
              TERMINAL_CWD: cwd,
              [TOKEN_ENV]: token
            },
            shell: backend.shell,
            stdio: ['ignore', 'pipe', 'pipe']
          })
        )

        sidecar = child
        let settled = false
        let stdout = ''

        const finish = (error: Error | null, value: { url: string; token: string } | null = null) => {
          if (settled) {
            return
          }

          settled = true
          clearTimeout(timer)
          sidecarStart = null

          if (error) {
            deps.stopBackendChild(child)
            reject(error)
          } else {
            resolve(value as { url: string; token: string })
          }
        }

        const timer = setTimeout(
          () => finish(new Error('Timed out waiting for local Computer Use bridge to announce its port.')),
          START_TIMEOUT_MS
        )

        child.stdout.on('data', chunk => {
          const text = String(chunk || '')
          stdout += text
          log(text.trim())
          const url = urlFromStdout(stdout)

          if (url) {
            sidecarInfo = { url, token }
            finish(null, sidecarInfo)
          }
        })
        child.stderr.on('data', chunk => log(String(chunk || '').trim()))
        child.once('error', error => finish(error))
        child.once('exit', (code, signal) => {
          log(`local sidecar exited (${signal || code})`)

          if (sidecar === child) {
            sidecar = null
            sidecarInfo = null
          }

          finish(new Error(`Local Computer Use bridge exited before ready (${signal || code}).`))
          lifecycle.closeSockets(null, true)
        })
      })
    })()

    return sidecarStart
  }

  function reply(ws: any, id: string, payload: Record<string, unknown>): void {
    if (!id || !ws || ws.readyState !== 1) {
      return
    }

    ws.send(JSON.stringify({ id, ...payload }))
  }

  async function serve(state: BridgeConnection, event: any): Promise<void> {
    let frame: any

    try {
      const raw = typeof event.data === 'string' ? event.data : Buffer.from(event.data).toString('utf8')
      frame = JSON.parse(raw)
    } catch (error: any) {
      log(`ignored invalid remote frame: ${error.message}`)

      return
    }

    const id = String(frame?.id || '')

    if (!id || lifecycle.connections.get(state.remoteKey) !== state) {
      return
    }

    try {
      if (frame.type === 'status') {
        const data = await request(state.url, state.token, '/v1/status')
        reply(state.ws, id, { ok: true, result: data?.status || data })

        return
      }

      if (frame.type === 'computer-use') {
        const data = await request(state.url, state.token, '/v1/computer-use', {
          method: 'POST',
          body: { method: frame.method, args: frame.args || {} }
        })

        reply(state.ws, id, { ok: true, result: data?.result })

        return
      }

      throw new Error(`Unsupported Computer Use bridge request type: ${frame.type}`)
    } catch (error: any) {
      reply(state.ws, id, { ok: false, error: error.message || String(error) })
    }
  }

  async function wsUrlFor(remote: RemoteTarget, profile: string | null): Promise<string> {
    if (remote.authMode === 'oauth') {
      const ticket = await deps.mintGatewayWsTicket(remote.baseUrl)

      return buildComputerUseBridgeWsUrlWithTicket(remote.baseUrl, ticket, profile)
    }

    return buildComputerUseBridgeWsUrl(remote.baseUrl, remote.token, profile)
  }

  function scheduleReconnect(remote: RemoteTarget, profile: string | null, remoteKey: string): void {
    if (
      stopping ||
      unsupportedReason ||
      !remote?.computerUseBridge ||
      !lifecycle.hasOwners(remoteKey) ||
      lifecycle.reconnectTimers.has(remoteKey)
    ) {
      return
    }

    const timer = setTimeout(() => {
      lifecycle.reconnectTimers.delete(remoteKey)

      if (!lifecycle.hasOwners(remoteKey)) {
        return
      }

      void ensure(remote, profile).catch(error => log(`reconnect failed: ${error.message || error}`))
    }, RECONNECT_MS)

    lifecycle.reconnectTimers.set(remoteKey, timer)
  }

  async function connect(
    remote: RemoteTarget,
    profile: string | null,
    remoteKey: string,
    capturedGeneration: number,
    scopedGeneration: number
  ): Promise<BridgeConnection | null> {
    const local = await ensureSidecar()

    if (stopping || capturedGeneration !== generation || !lifecycle.isCurrent(remoteKey, scopedGeneration)) {
      return null
    }

    const existing = lifecycle.connections.get(remoteKey)

    if (existing?.ws && [0, 1].includes(existing.ws.readyState)) {
      return existing
    }

    lifecycle.closeSockets(remoteKey)
    const wsUrl = await wsUrlFor(remote, profile)

    if (stopping || capturedGeneration !== generation || !lifecycle.isCurrent(remoteKey, scopedGeneration)) {
      return null
    }

    const WebSocketImpl = deps.webSocketImpl || (globalThis as any).WebSocket
    const ws = new WebSocketImpl(wsUrl)

    const state: BridgeConnection = {
      ...local,
      remoteKey,
      remoteBaseUrl: remote.baseUrl,
      profile: connectionScopeKey(profile),
      ws
    }

    lifecycle.connections.set(remoteKey, state)

    await new Promise<void>((resolve, reject) => {
      let settled = false

      const finish = (error?: Error) => {
        if (settled) {
          return
        }

        settled = true
        clearTimeout(timer)

        if (error) {
          reject(error)
        } else {
          resolve()
        }
      }

      const timer = setTimeout(
        () => finish(new Error('Timed out connecting local Computer Use bridge to remote backend.')),
        CONNECT_TIMEOUT_MS
      )

      ws.addEventListener('open', () => {
        log(`connected to ${remote.baseUrl} for profile "${state.profile || 'current'}"`)
        finish()
      })
      ws.addEventListener('message', event => void serve(state, event))
      ws.addEventListener('error', (event: any) => {
        log(`websocket error: ${event?.message || 'connection failed'}`)
        finish(new Error('Computer Use bridge WebSocket connection failed.'))
      })
      ws.addEventListener('close', (event: any) => {
        log(`websocket closed (${event?.code || 'unknown'})`)
        const closedByDesktop = lifecycle.connections.get(remoteKey)?.closedByDesktop

        if (lifecycle.connections.get(remoteKey)?.ws === ws) {
          lifecycle.connections.delete(remoteKey)
        }

        finish(new Error('Computer Use bridge WebSocket closed before it became ready.'))

        if (!closedByDesktop) {
          scheduleReconnect(remote, profile, remoteKey)
        }
      })
    })

    return state
  }

  /** Open (or reuse) this remote+profile's bridge. Returns null when it is off. */
  async function ensure(remote: RemoteTarget, profile: string | null = null): Promise<BridgeConnection | null> {
    if (!remote?.computerUseBridge) {
      return null
    }

    if (unsupported()) {
      log(unsupportedReason as string)

      return null
    }

    if (typeof (deps.webSocketImpl || (globalThis as any).WebSocket) !== 'function') {
      log('WebSocket is not available in this Electron runtime; local bridge disabled.')

      return null
    }

    stopping = false
    const scopedProfile = remote.source === 'profile' ? null : profile
    const remoteKey = remoteKeyFor(remote, scopedProfile)

    if (!lifecycle.hasOwners(remoteKey)) {
      return null
    }

    const existing = lifecycle.connections.get(remoteKey)

    if (existing?.ws && [0, 1].includes(existing.ws.readyState)) {
      return existing
    }

    const pending = lifecycle.connectionPromises.get(remoteKey)

    if (pending) {
      return pending
    }

    const capturedGeneration = generation
    const scopedGeneration = lifecycle.generation(remoteKey)
    const attempt = connect(remote, scopedProfile, remoteKey, capturedGeneration, scopedGeneration)
    lifecycle.connectionPromises.set(remoteKey, attempt)

    try {
      return await attempt
    } catch (error: any) {
      scheduleBridgeReconnectIfCurrent({
        lifecycle,
        remoteKey,
        stopping,
        enabled: Boolean(remote.computerUseBridge) && !unsupportedReason,
        capturedGlobalGeneration: capturedGeneration,
        currentGlobalGeneration: generation,
        capturedScopedGeneration: scopedGeneration,
        scheduleReconnect: () => scheduleReconnect(remote, scopedProfile, remoteKey)
      })
      throw error
    } finally {
      if (lifecycle.connectionPromises.get(remoteKey) === attempt) {
        lifecycle.connectionPromises.delete(remoteKey)
      }
    }
  }

  /** Claim this remote+profile for an owner, so its socket outlives one attempt. */
  function acquire(remote: RemoteTarget, profile: string | null, owner: string): string {
    const remoteKey = remoteKeyFor(remote, remote.source === 'profile' ? null : profile)
    lifecycle.acquire(remoteKey, owner)

    return remoteKey
  }

  /** The app's own connection, which replaces whatever it pointed at before. */
  async function ensurePrimary(remote: RemoteTarget | null, profile: string | null = null): Promise<void> {
    const previous = primaryRemoteKey

    if (!remote?.computerUseBridge) {
      primaryRemoteKey = null

      if (previous) {
        release(previous, PRIMARY_OWNER)
      }

      return
    }

    const remoteKey = acquire(remote, profile, PRIMARY_OWNER)

    if (previous && previous !== remoteKey) {
      release(previous, PRIMARY_OWNER)
    }

    primaryRemoteKey = remoteKey

    await ensure(remote, profile).catch(error => log(`setup failed: ${error.message || error}`))
  }

  function release(remoteKey: string, owner: string): boolean {
    return releaseBridgeOwnerAndStopSidecarIfIdle({ lifecycle, remoteKey, owner, stopSidecar })
  }

  function stopAll(): void {
    stopping = true
    generation += 1
    primaryRemoteKey = null
    lifecycle.cancelAll()
    stopSidecar()
  }

  return {
    acquire,
    ensure,
    ensurePrimary,
    release,
    stopAll,
    unsupported,
    PRIMARY_OWNER
  }
}

export type ComputerUseBridge = ReturnType<typeof createComputerUseBridge>
