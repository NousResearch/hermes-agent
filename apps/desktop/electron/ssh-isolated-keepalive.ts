/**
 * Long-lived keep-alive WebSocket from Electron main for every published
 * SSH-isolated backend (#106935).
 *
 * `web_server_idle_exit` (#101626) treats accepted WebSockets as ownership
 * liveness. Renderer sockets can vanish (`disposeSecondary` /
 * `pruneSecondaryGateways`) while Desktop still owns the backend via
 * `sshConnections`. Sticky spawn artifacts (owner-nonce, token file, lockfile)
 * are NOT liveness and must not suppress idle-exit.
 *
 * If the socket drops the registry reconnects with capped exponential backoff;
 * while it is down the backend may idle-exit as before. This module never consults
 * nonce/lock/token files.
 */
import { buildGatewayWsUrl } from './connection-config'

export type SshIsolatedKeepaliveTarget = {
  baseUrl: string
  token: string
}

export type SshIsolatedKeepaliveOptions = {
  WebSocketImpl?: any
  buildWsUrl?: (baseUrl: string, token: string) => string
  log?: (message: string) => void
  reconnectDelayMs?: number
}

type KeepaliveEntry = {
  failures: number
  reconnectTimer: ReturnType<typeof setTimeout> | null
  scope: string
  socket: { close?: () => void; url?: string } | null
  target: SshIsolatedKeepaliveTarget
}

const DEFAULT_RECONNECT_DELAY_MS = 2_000
const MAX_RECONNECT_DELAY_MS = 30_000

function addListener(socket: any, type: string, handler: (event?: any) => void) {
  if (typeof socket?.addEventListener === 'function') {
    socket.addEventListener(type, handler)

    return
  }

  if (typeof socket?.on === 'function') {
    socket.on(type, handler)
  }
}

export function createSshIsolatedKeepaliveRegistry(options: SshIsolatedKeepaliveOptions = {}) {
  const WebSocketImpl =
    'WebSocketImpl' in options ? options.WebSocketImpl : (globalThis as { WebSocket?: unknown }).WebSocket

  const buildWsUrl = options.buildWsUrl ?? buildGatewayWsUrl
  const log = options.log
  const reconnectDelayMs = options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS
  const entries = new Map<string, KeepaliveEntry>()

  function clearTimer(entry: KeepaliveEntry) {
    if (entry.reconnectTimer == null) {
      return
    }

    clearTimeout(entry.reconnectTimer)
    entry.reconnectTimer = null
  }

  function closeSocket(entry: KeepaliveEntry) {
    const socket = entry.socket
    entry.socket = null

    if (!socket) {
      return
    }

    try {
      socket.close?.()
    } catch {
      // Best-effort teardown; a dead socket must not block scope cleanup.
    }
  }

  function scheduleReconnect(entry: KeepaliveEntry) {
    if (entries.get(entry.scope) !== entry || entry.reconnectTimer != null) {
      return
    }

    // A dead tunnel would otherwise be redialled every 2 s until the scope is torn down.
    const delay = Math.min(reconnectDelayMs * 2 ** entry.failures, MAX_RECONNECT_DELAY_MS)
    entry.failures += 1
    entry.reconnectTimer = setTimeout(() => {
      entry.reconnectTimer = null
      connect(entry)
    }, delay)
  }

  function connect(entry: KeepaliveEntry) {
    if (entries.get(entry.scope) !== entry) {
      return
    }

    clearTimer(entry)
    closeSocket(entry)

    let socket: any
    let url: string

    try {
      url = buildWsUrl(entry.target.baseUrl, entry.target.token)
      socket = new WebSocketImpl(url)
    } catch (error) {
      log?.(`[ssh] keep-alive WS failed to open for ${entry.scope}: ${error instanceof Error ? error.message : error}`)
      scheduleReconnect(entry)

      return
    }

    entry.socket = socket

    // Staleness is derivable: stop() removes the entry, connect() replaces entry.socket.
    const abandonIfStale = () => {
      if (entries.get(entry.scope) !== entry || entry.socket !== socket) {
        return
      }

      entry.socket = null
      scheduleReconnect(entry)
    }

    addListener(socket, 'open', () => {
      if (entry.socket === socket) {
        entry.failures = 0
      }
    })
    addListener(socket, 'close', abandonIfStale)
    addListener(socket, 'error', abandonIfStale)
  }

  function start(scope: string, target: SshIsolatedKeepaliveTarget) {
    // '' is a real published key: sshScopeKey(null) for the v1/global SSH primary.
    if (typeof scope !== 'string') {
      return
    }

    const baseUrl = typeof target?.baseUrl === 'string' ? target.baseUrl : ''
    const token = typeof target?.token === 'string' ? target.token : ''

    if (!baseUrl || !token) {
      return
    }

    stop(scope)

    const entry: KeepaliveEntry = {
      failures: 0,
      reconnectTimer: null,
      scope,
      socket: null,
      target: { baseUrl, token }
    }

    entries.set(scope, entry)
    connect(entry)
  }

  function stop(scope: string) {
    const entry = entries.get(scope)

    if (!entry) {
      return
    }

    entries.delete(scope)
    clearTimer(entry)
    closeSocket(entry)
  }

  function stopAll() {
    for (const scope of [...entries.keys()]) {
      stop(scope)
    }
  }

  function isArmed(scope: string) {
    return entries.has(scope)
  }

  function openUrl(scope: string) {
    const url = entries.get(scope)?.socket?.url

    return typeof url === 'string' ? url : null
  }

  return { isArmed, openUrl, start, stop, stopAll }
}
