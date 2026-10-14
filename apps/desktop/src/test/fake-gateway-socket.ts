type Listener = (ev: unknown) => void

// Minimal WebSocket stand-in implementing only what json-rpc-gateway.connect()
// touches: readyState, add/removeEventListener('open'|'error'|'close'), close().
export class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  // Flipped by the test: 'open' = next socket connects; 'fail' = next socket
  // errors (a dead remote). Mirrors a VPS going away after the first connect.
  static mode: 'open' | 'fail' = 'open'
  static instances: FakeWebSocket[] = []
  // Ping behavior: 'pong' answers with a healthy pong frame; 'silent' swallows
  // the request (the half-open-socket simulation — connection looks OPEN but
  // every RPC hangs until its per-call timeout); 'method-not-found' answers
  // the JSON-RPC error a PRE-ping backend returns (a healthy, version-skewed
  // response that must NOT trigger a reconnect).
  static pingMode: 'pong' | 'silent' | 'method-not-found' = 'pong'

  readyState = 0
  private listeners: Record<string, Set<Listener>> = {}

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
    const willOpen = FakeWebSocket.mode === 'open'
    // Resolve on the next microtask/macrotask so connect()'s promise wiring is
    // in place before open/error fires (matches real async socket handshake).
    setTimeout(() => {
      if (willOpen) {
        this.readyState = FakeWebSocket.OPEN
        this.emit('open', {})
      } else {
        this.readyState = FakeWebSocket.CLOSED
        this.emit('error', {})
      }
    }, 0)
  }

  addEventListener(type: string, fn: Listener) {
    ;(this.listeners[type] ??= new Set()).add(fn)
  }

  removeEventListener(type: string, fn: Listener) {
    this.listeners[type]?.delete(fn)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close', {})
  }

  // Force-drop an open socket, as a sleeping laptop / restarted remote would.
  drop() {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close', {})
  }

  send(data: string) {
    let frame: { id?: unknown; method?: string }

    try {
      frame = JSON.parse(data) as { id?: unknown; method?: string }
    } catch {
      return
    }

    if (frame.method !== 'ping') {
      return
    }

    if (FakeWebSocket.pingMode === 'pong') {
      this.emit('message', {
        data: JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { pong: true } })
      })
    } else if (FakeWebSocket.pingMode === 'method-not-found') {
      this.emit('message', {
        data: JSON.stringify({
          jsonrpc: '2.0',
          id: frame.id,
          error: { code: -32601, message: 'Method not found' }
        })
      })
    }
    // 'silent': swallow — a healthy socket answers, a half-open one never does.
  }

  private emit(type: string, ev: unknown) {
    for (const fn of this.listeners[type] ?? []) {
      fn(ev)
    }
  }
}
