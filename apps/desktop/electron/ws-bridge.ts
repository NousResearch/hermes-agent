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
import { ipcMain } from 'electron'

import WebSocket from 'ws'

interface BridgeSocket {
  ws: InstanceType<typeof WebSocket>
  sender: Electron.WebContents
}

const sockets = new Map<number, BridgeSocket>()
let nextId = 1

const CHANNEL_EVENT = 'hermes:ws-bridge:event'

export function installWebSocketBridge(): void {
  ipcMain.handle('hermes:ws-bridge:open', (event, url: string) => {
    return new Promise(resolve => {
      const id = nextId++
      let settled = false
      let ws: InstanceType<typeof WebSocket>
      try {
        ws = new WebSocket(url, { maxPayload: 64 * 1024 * 1024 })
      } catch (err) {
        resolve({ ok: false, error: String(err) })
        return
      }

      const send = (payload: unknown) => {
        if (!event.sender.isDestroyed()) {
          event.sender.send(CHANNEL_EVENT, id, payload)
        }
      }

      ws.on('open', () => {
        sockets.set(id, { ws, sender: event.sender })
        // Register the socket, THEN resolve so the renderer learns the id,
        // THEN emit open — any ordering where 'open' could fire before the
        // renderer knows its id drops the event and hangs the client in
        // 'connecting' forever.
        if (!settled) {
          settled = true
          resolve({ ok: true, id })
        }
        // Defer past the renderer's promise microtask so id assignment lands
        // before the open event is matched.
        setImmediate(() => send({ type: 'open' }))
      })
      ws.on('message', (data: Buffer | string, isBinary: boolean) => {
        send({ type: 'message', data: isBinary ? data.toString('base64') : String(data), binary: isBinary })
      })
      ws.on('error', (err: Error) => {
        send({ type: 'error', message: err.message })
      })
      ws.on('close', (code: number, reason: Buffer) => {
        sockets.delete(id)
        send({ type: 'close', code, reason: reason.toString() })
        if (!settled) {
          settled = true
          resolve({ ok: false, error: `WebSocket closed during connect (code ${code})` })
        }
      })
      // Safety: if neither open nor close lands, don't hang the renderer dial.
      setTimeout(() => {
        if (!settled) {
          settled = true
          try { ws.terminate() } catch { /* already gone */ }
          resolve({ ok: false, error: 'WebSocket connect timed out' })
        }
      }, 20_000)
    })
  })

  ipcMain.handle('hermes:ws-bridge:send', (_event, id: number, data: string, binary: boolean) => {
    const entry = sockets.get(id)
    if (!entry || entry.ws.readyState !== entry.ws.OPEN) return { ok: false }
    entry.ws.send(binary ? Buffer.from(data, 'base64') : data)
    return { ok: true }
  })

  ipcMain.handle('hermes:ws-bridge:close', (_event, id: number, code?: number, reason?: string) => {
    const entry = sockets.get(id)
    if (entry) {
      sockets.delete(id)
      try { entry.ws.close(code, reason) } catch { /* already gone */ }
    }
    return { ok: true }
  })
}
