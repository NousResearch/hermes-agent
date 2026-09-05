// Renderer side of the remote WebSocket bridge (see electron/ws-bridge.ts).
// socketFactory for JsonRpcGatewayClient: wss:// dials are piped through
// Electron's main process (`ws` npm package, node:tls — honors
// NODE_EXTRA_CA_CERTS); ws:// loopback keeps the native WebSocket.
//
// The client type is `WebSocketLike = WebSocket` and it drives sockets via
// addEventListener('open'|'message'|'close'|'error') plus send/close/
// readyState — BridgedWebSocket mirrors that EventTarget surface exactly.
interface BridgeApi {
  wsBridgeOpen: (url: string) => Promise<{ ok: boolean; id?: number; error?: string }>
  wsBridgeSend: (id: number, data: string, binary: boolean) => Promise<{ ok: boolean }>
  wsBridgeClose: (id: number, code?: number, reason?: string) => Promise<{ ok: boolean }>
  onWsBridgeEvent: (
    callback: (id: number, payload: { type: string; data?: string; binary?: boolean; code?: number; reason?: string }) => void
  ) => () => void
}

function bridgeApi(): BridgeApi | null {
  const api = (window as unknown as { hermesDesktop?: Partial<BridgeApi> }).hermesDesktop
  return api && typeof api.wsBridgeOpen === 'function' ? (api as BridgeApi) : null
}

class BridgedWebSocket extends EventTarget {
  readonly CONNECTING = 0
  readonly OPEN = 1
  readonly CLOSING = 2
  readonly CLOSED = 3

  readyState = 0
  binaryType: 'blob' | 'arraybuffer' = 'arraybuffer'

  private id: number | null = null
  private readonly removeListener: () => void
  private pending: Array<{ type: string; data?: string; binary?: boolean; code?: number; reason?: string }> = []

  constructor(url: string, api: BridgeApi) {
    super()
    this.removeListener = api.onWsBridgeEvent((id, payload) => {
      if (this.id === null) {
        // Event arrived before wsBridgeOpen resolved and assigned our id.
        // Buffer it — flushed in order once the id lands. Without this an
        // 'open' racing the id assignment is dropped and the client hangs
        // in 'connecting' forever.
        this.pending.push(payload)
        return
      }
      if (id !== this.id) return
      this.dispatch(payload)
    })

    void api.wsBridgeOpen(url).then(result => {
      if (!result.ok || result.id === undefined) {
        this.readyState = 3
        this.dispatchEvent(new Event('error'))
        this.dispatchEvent(new CloseEvent('close', { code: 1006, reason: result.error ?? 'bridge dial failed' }))
        return
      }
      this.id = result.id
      for (const payload of this.pending) this.dispatch(payload)
      this.pending = []
    })
  }

  private dispatch(payload: { type: string; data?: string; binary?: boolean; code?: number; reason?: string }): void {
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
        this.readyState = 3
        this.dispatchEvent(new CloseEvent('close', { code: payload.code ?? 1006, reason: payload.reason ?? '' }))
        break
    }
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    if (this.id === null || this.readyState !== 1) return
    const api = bridgeApi()
    if (!api) return
    if (typeof data === 'string') {
      void api.wsBridgeSend(this.id, data, false)
      return
    }
    if (data instanceof ArrayBuffer) {
      void api.wsBridgeSend(this.id, arrayBufferToBase64(data), true)
      return
    }
    if (ArrayBuffer.isView(data)) {
      void api.wsBridgeSend(this.id, arrayBufferToBase64(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer), true)
      return
    }
    // Blob: async read then send.
    void (data as Blob).arrayBuffer().then((buf: ArrayBuffer) => {
      if (this.id !== null && this.readyState === 1) void api.wsBridgeSend(this.id, arrayBufferToBase64(buf), true)
    })
  }

  close(code?: number, reason?: string): void {
    if (this.readyState >= 2) return
    this.readyState = 2
    this.removeListener()
    if (this.id !== null) {
      const api = bridgeApi()
      if (api) void api.wsBridgeClose(this.id, code, reason)
    }
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
