// Renderer side of the remote WebSocket bridge (see electron/ws-bridge.ts).
// socketFactory for JsonRpcGatewayClient: wss:// dials are piped through
// Electron's main process (`ws` npm package, node:tls — honors
// NODE_EXTRA_CA_CERTS); ws:// loopback keeps the native WebSocket.
//
// The client type is `WebSocketLike = WebSocket` and it drives sockets via
// addEventListener('open'|'message'|'close'|'error') plus send/close/
// readyState — BridgedWebSocket mirrors that EventTarget surface exactly.
//
// Concurrency/lifecycle invariants (multiple bridged sockets per renderer are
// NORMAL — primary + per-profile secondaries):
//   - Every dial carries a client-generated token; events arrive tagged by
//     token, so socket A's frames can never replay into socket B.
//   - close() while CONNECTING cancels the main-process dial by token —
//     a canceled token is never promoted into main's live map.
//   - The IPC listener is removed on EVERY terminal outcome (open-fail,
//     remote close, local close) — reconnect cycles don't accumulate them.
interface BridgeApi {
  wsBridgeOpen: (url: string, token: string) => Promise<{ ok: boolean; error?: string }>
  wsBridgeCancel: (token: string) => Promise<{ ok: boolean }>
  wsBridgeSend: (token: string, data: string, binary: boolean) => Promise<{ ok: boolean }>
  wsBridgeClose: (token: string, code?: number, reason?: string) => Promise<{ ok: boolean }>
  onWsBridgeEvent: (
    callback: (token: string, payload: BridgePayload) => void
  ) => () => void
}

interface BridgePayload {
  type: string
  data?: string
  binary?: boolean
  code?: number
  reason?: string
}

function bridgeApi(): BridgeApi | null {
  const api = (window as unknown as { hermesDesktop?: Partial<BridgeApi> }).hermesDesktop
  return api && typeof api.wsBridgeOpen === 'function' ? (api as BridgeApi) : null
}

let nextToken = 1

export class BridgedWebSocket extends EventTarget {
  readonly CONNECTING = 0
  readonly OPEN = 1
  readonly CLOSING = 2
  readonly CLOSED = 3

  readyState = 0
  binaryType: 'blob' | 'arraybuffer' = 'arraybuffer'

  private readonly token: string
  private removeListener: () => void
  private terminated = false

  constructor(url: string, private readonly api: BridgeApi, token?: string) {
    super()
    this.token = token ?? `dial-${nextToken++}-${Math.random().toString(36).slice(2)}`
    this.removeListener = api.onWsBridgeEvent((token, payload) => {
      // Events are token-tagged end to end — unrelated sockets' frames are
      // rejected from the start, including anything arriving before open
      // resolved (the bridge emits open only after resolving, so ordering
      // is: open-result promise, then events, all under our token).
      if (token !== this.token) return
      this.dispatch(payload)
    })

    void api.wsBridgeOpen(url, this.token).then(result => {
      if (this.terminated) {
        // close() raced the dial resolution: if it opened anyway, shut it.
        if (result.ok) void this.api.wsBridgeClose(this.token)
        return
      }
      if (!result.ok) {
        this.terminate()
        this.readyState = 3
        this.dispatchEvent(new Event('error'))
        this.dispatchEvent(new CloseEvent('close', { code: 1006, reason: result.error ?? 'bridge dial failed' }))
      }
      // On success the 'open' event arrives over the channel (deferred past
      // this microtask by the bridge) and flips readyState in dispatch().
    })
  }

  private terminate(): void {
    if (this.terminated) return
    this.terminated = true
    this.removeListener()
  }

  private dispatch(payload: BridgePayload): void {
    switch (payload.type) {
      case 'open':
        this.readyState = 1
        this.dispatchEvent(new Event('open'))
        break
      case 'message':
        this.dispatchEvent(
          new MessageEvent('message', {
            data: payload.binary ? base64ToArrayBuffer(payload.data ?? '') : (payload.data ?? '')
          })
        )
        break
      case 'error':
        this.dispatchEvent(new Event('error'))
        break
      case 'close':
        this.terminate()
        this.readyState = 3
        this.dispatchEvent(new CloseEvent('close', { code: payload.code ?? 1006, reason: payload.reason ?? '' }))
        break
    }
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    if (this.readyState !== 1 || this.terminated) return
    if (typeof data === 'string') {
      void this.api.wsBridgeSend(this.token, data, false)
      return
    }
    if (data instanceof ArrayBuffer) {
      void this.api.wsBridgeSend(this.token, arrayBufferToBase64(data), true)
      return
    }
    if (ArrayBuffer.isView(data)) {
      void this.api.wsBridgeSend(this.token, arrayBufferToBase64(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer), true)
      return
    }
    // Blob: async read then send.
    void (data as Blob).arrayBuffer().then((buf: ArrayBuffer) => {
      if (this.readyState === 1 && !this.terminated) void this.api.wsBridgeSend(this.token, arrayBufferToBase64(buf), true)
    })
  }

  close(code?: number, reason?: string): void {
    if (this.readyState >= 2) return
    this.readyState = 2
    this.terminate()
    // Cancel covers a still-dialing token; close covers an established one —
    // main no-ops whichever half doesn't apply, and enforces sender ownership.
    void this.api.wsBridgeCancel(this.token)
    void this.api.wsBridgeClose(this.token, code, reason)
    this.readyState = 3
  }
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return bytes.buffer
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let bin = ''
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
  return btoa(bin)
}

/** socketFactory for JsonRpcGatewayClient: bridge wss:// through the main
 *  process (private-CA trust), keep native WebSocket for cleartext loopback. */
export function gatewaySocketFactory(url: string): WebSocket {
  const api = bridgeApi()
  if (api && url.startsWith('wss://')) {
    return new BridgedWebSocket(url, api) as unknown as WebSocket
  }
  return new WebSocket(url)
}
