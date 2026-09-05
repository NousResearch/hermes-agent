// Remote WebSocket bridge: dials remote (wss://) gateway sockets from Electron's
// MAIN process using the `ws` npm package, which honors NODE_EXTRA_CA_CERTS.
//
// Why this exists: two TLS clients in the app do NOT trust private CAs the way
// the rest of the stack does —
//   1. Chromium's WebSocket in the renderer ignores --use-system-certificates
//      on Linux (URL loaders use the system store; the WS socket pool does not).
//   2. Node's built-in undici WebSocket ignores NODE_EXTRA_CA_CERTS entirely.
// The `ws` package uses node:tls, which honors NODE_EXTRA_CA_CERTS — verified
// against a NullX-chain host: handshake completes, auth layer reachable.
//
// The bridge is a dumb frame pipe: the renderer gets a WebSocket-like object
// whose send/close are IPC invokes and whose events arrive over a per-socket
// channel. Local ws://127.0.0.1 dials keep the native renderer WebSocket —
// no TLS involved, no behavior change.
//
// Authority rules (moving the socket across the process boundary creates new
// edges — these are the invariants that keep it safe):
//   - Headers come from the MAIN-OWNED store (same resolution as the renderer
//     webRequest path), never from renderer-supplied input.
//   - The renderer supplies a dial token with open; every mutation (cancel/
//     send/close) must present it, and every socket is additionally owned by
//     the invoking WebContents. A renderer can never address another
//     renderer's socket, and never a dial it didn't start.
//   - A canceled dial token is never promoted into the live map, even if the
//     underlying ws opens late (client connect timeout is 15s — one layer
//     owns the deadline).
//   - Every socket/dial owned by a destroyed WebContents is torn down with it.
import type { IpcMain, WebContents } from 'electron'

import WebSocket from 'ws'

// `electron` is a type-only import: the runtime value (ipcMain) is injected
// through deps so this module loads under bare node:test. installWebSocketBridge
// resolves the real ipcMain lazily via require at call time (Electron main only).

interface LiveSocket {
  ws: WebSocketLike
  sender: WebContents
}

interface PendingDial {
  ws: WebSocketLike
  sender: WebContents
  token: string
  canceled: boolean
  watchdog: ReturnType<typeof setTimeout>
}

export interface WsLike {
  readyState: number
  send(data: unknown): void
  close(code?: number, reason?: string): void
  terminate(): void
  on(event: 'open', fn: () => void): void
  on(event: 'message', fn: (data: Buffer | string, isBinary: boolean) => void): void
  on(event: 'error', fn: (err: Error) => void): void
  on(event: 'close', fn: (code: number, reason: Buffer) => void): void
}

type WebSocketLike = WsLike

export interface WebSocketBridgeDeps {
  /** Main-owned, sanitized per-URL header resolution — the same source the
   *  renderer webRequest.onBeforeSendHeaders path uses. */
  headersForUrl?: (url: string) => Record<string, string>
  /** DI seams for tests. */
  ipc?: Pick<IpcMain, 'handle'>
  webSocketImpl?: new (url: string, options?: { headers?: Record<string, string>; maxPayload?: number }) => WebSocketLike
  /** Connect deadline. Default matches DEFAULT_CONNECT_TIMEOUT_MS in
   *  apps/shared/src/json-rpc-gateway.ts — one layer must own the deadline or
   *  a late open can be promoted after the client gave up on the socket. */
  connectTimeoutMs?: number
}

const CHANNEL_EVENT = 'hermes:ws-bridge:event'

function defaultIpcMain(): Pick<IpcMain, 'handle'> {
  // Lazy require keeps `electron` out of the module graph under node:test.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return (require('electron') as typeof import('electron')).ipcMain
}

export function createWebSocketBridge(deps: WebSocketBridgeDeps = {}) {
  const headersForUrl = deps.headersForUrl ?? (() => ({}))
  const ipc = deps.ipc ?? defaultIpcMain()
  const WsImpl = deps.webSocketImpl ?? (WebSocket as unknown as WebSocketBridgeDeps['webSocketImpl'])!
  const connectTimeoutMs = deps.connectTimeoutMs ?? 15_000

  const sockets = new Map<string, LiveSocket>()
  const pendingDials = new Map<string, PendingDial>()

  const sendTo = (sender: WebContents, token: string, payload: unknown) => {
    if (!sender.isDestroyed()) {
      sender.send(CHANNEL_EVENT, token, payload)
    }
  }

  const retireOwned = (sender: WebContents) => {
    for (const [token, entry] of sockets) {
      if (entry.sender === sender) {
        sockets.delete(token)
        try { entry.ws.terminate() } catch { /* already gone */ }
      }
    }
    for (const [token, dial] of pendingDials) {
      if (dial.sender === sender) {
        dial.canceled = true
        clearTimeout(dial.watchdog)
        pendingDials.delete(token)
        try { dial.ws.terminate() } catch { /* already gone */ }
      }
    }
  }

  function install(): void {
    ipc.handle('hermes:ws-bridge:open', (event, url: string, token: string) => {
      return new Promise(resolve => {
        const sender = event.sender
        if (typeof token !== 'string' || token.length === 0 || pendingDials.has(token) || sockets.has(token)) {
          resolve({ ok: false, error: 'invalid dial token' })
          return
        }
        let ws: WebSocketLike
        try {
          ws = new WsImpl(url, {
            headers: headersForUrl(url),
            maxPayload: 64 * 1024 * 1024
          })
        } catch (err) {
          resolve({ ok: false, error: String(err) })
          return
        }

        const dial: PendingDial = {
          ws,
          sender,
          token,
          canceled: false,
          watchdog: setTimeout(() => {
            if (pendingDials.delete(token)) {
              dial.canceled = true
              try { ws.terminate() } catch { /* already gone */ }
              resolve({ ok: false, error: 'WebSocket connect timed out' })
            }
          }, connectTimeoutMs)
        }
        pendingDials.set(token, dial)

        sender.once('destroyed', () => retireOwned(sender))

        ws.on('open', () => {
          if (dial.canceled || !pendingDials.delete(token)) {
            // Renderer gave up before the dial completed — never promote.
            try { ws.terminate() } catch { /* already gone */ }
            return
          }
          clearTimeout(dial.watchdog)
          sockets.set(token, { ws, sender })
          // Resolve BEFORE emitting open, and defer the open event past the
          // renderer's promise microtask so its bookkeeping lands first —
          // either race alone hangs the client in 'connecting' forever.
          resolve({ ok: true })
          setImmediate(() => sendTo(sender, token, { type: 'open' }))
        })
        ws.on('message', (data: Buffer | string, isBinary: boolean) => {
          sendTo(sender, token, { type: 'message', data: isBinary ? data.toString('base64') : String(data), binary: isBinary })
        })
        ws.on('error', (err: Error) => {
          sendTo(sender, token, { type: 'error', message: err.message })
        })
        ws.on('close', (code: number, reason: Buffer) => {
          clearTimeout(dial.watchdog)
          const wasPending = pendingDials.delete(token)
          sockets.delete(token)
          sendTo(sender, token, { type: 'close', code, reason: reason.toString() })
          if (wasPending && !dial.canceled) {
            resolve({ ok: false, error: `WebSocket closed during connect (code ${code})` })
          }
        })
      })
    })

    // Cancel a CONNECTING dial (renderer connect timeout fired before open).
    ipc.handle('hermes:ws-bridge:cancel', (event, token: string) => {
      const dial = pendingDials.get(token)
      if (!dial || dial.sender !== event.sender) return { ok: false }
      dial.canceled = true
      clearTimeout(dial.watchdog)
      pendingDials.delete(token)
      try { dial.ws.terminate() } catch { /* already gone */ }
      return { ok: true }
    })

    ipc.handle('hermes:ws-bridge:send', (event, token: string, data: string, binary: boolean) => {
      const entry = sockets.get(token)
      if (!entry || entry.sender !== event.sender) return { ok: false }
      if (entry.ws.readyState !== 1) return { ok: false }
      entry.ws.send(binary ? Buffer.from(data, 'base64') : data)
      return { ok: true }
    })

    ipc.handle('hermes:ws-bridge:close', (event, token: string, code?: number, reason?: string) => {
      const entry = sockets.get(token)
      if (!entry || entry.sender !== event.sender) return { ok: false }
      sockets.delete(token)
      try { entry.ws.close(code, reason) } catch { /* already gone */ }
      return { ok: true }
    })
  }

  return { install, sockets, pendingDials }
}

export function installWebSocketBridge(deps: WebSocketBridgeDeps = {}): void {
  createWebSocketBridge(deps).install()
}
