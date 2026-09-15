import { shouldDialGatewayFromMain } from './gateway-ws-target'

type GatewayWsEvent =
  | { data?: string; id: string; type: 'open' }
  | { data?: string; id: string; type: 'message' }
  | { code?: number; id: string; reason?: string; type: 'close' }
  | { id: string; type: 'error' }

export interface GatewayWsBridge {
  close: (id: string) => void
  open: (url: string) => Promise<{ error?: string; id?: string; ok: boolean }>
  send: (id: string, data: string) => void
  subscribe: (callback: (event: GatewayWsEvent) => void) => () => void
}

const CONNECTING = 0
const OPEN = 1
const CLOSING = 2
const CLOSED = 3

function gatewayWsBridge(): GatewayWsBridge | null {
  if (typeof window === 'undefined') {
    return null
  }

  return window.hermesDesktop?.gatewayWs ?? null
}

/**
 * Chromium WebSocket for loopback; Node/undici WebSocket (via main) for remote
 * gateways — the same transport "Test remote" already proved works.
 */
export function createDesktopGatewaySocket(url: string, bridge = gatewayWsBridge()): WebSocket {
  if (bridge && shouldDialGatewayFromMain(url)) {
    return new DesktopMainGatewaySocket(url, bridge) as unknown as WebSocket
  }

  return new WebSocket(url)
}

export class DesktopMainGatewaySocket {
  readyState = CONNECTING
  url: string
  private readonly listeners = {
    close: new Set<(event: CloseEvent) => void>(),
    error: new Set<(event: Event) => void>(),
    message: new Set<(event: MessageEvent) => void>(),
    open: new Set<(event: Event) => void>()
  }
  private id: string | null = null
  private readonly pending: GatewayWsEvent[] = []
  private readonly unsubscribe: () => void

  constructor(
    url: string,
    private readonly bridge: GatewayWsBridge
  ) {
    this.url = url
    this.unsubscribe = bridge.subscribe(event => this.dispatch(event))
    void bridge.open(url).then(
      result => {
        if (!result.ok || !result.id) {
          this.fail()

          return
        }

        this.id = result.id

        for (const event of this.pending.splice(0)) {
          this.dispatch(event)
        }
      },
      () => this.fail()
    )
  }

  addEventListener(
    type: keyof DesktopMainGatewaySocket['listeners'],
    listener: (event: any) => void,
    options?: { once?: boolean }
  ): void {
    const bucket = this.listeners[type]

    if (!bucket) {
      return
    }

    const wrapped = options?.once
      ? (event: any) => {
          bucket.delete(wrapped)
          listener(event)
        }
      : listener

    bucket.add(wrapped)
  }

  removeEventListener(type: keyof DesktopMainGatewaySocket['listeners'], listener: (event: any) => void): void {
    this.listeners[type]?.delete(listener)
  }

  send(data: string): void {
    if (this.id && this.readyState === OPEN) {
      this.bridge.send(this.id, data)
    }
  }

  close(): void {
    if (this.readyState === CLOSED || this.readyState === CLOSING) {
      return
    }

    this.readyState = CLOSING

    if (this.id) {
      this.bridge.close(this.id)
    }
  }

  private dispatch(event: GatewayWsEvent): void {
    if (this.id && event.id !== this.id) {
      return
    }

    if (!this.id) {
      this.pending.push(event)

      return
    }

    if (event.type === 'open') {
      this.readyState = OPEN
      this.emit('open', new Event('open'))

      return
    }

    if (event.type === 'message') {
      this.emit('message', { data: event.data ?? '' } as MessageEvent)

      return
    }

    if (event.type === 'error') {
      this.readyState = CLOSED
      this.emit('error', new Event('error'))
      this.unsubscribe()

      return
    }

    this.readyState = CLOSED
    this.emit('close', { code: event.code ?? 1005, reason: event.reason ?? '' } as CloseEvent)
    this.unsubscribe()
  }

  private fail(): void {
    if (this.readyState === CLOSED) {
      return
    }

    this.readyState = CLOSED
    this.emit('error', new Event('error'))
    this.unsubscribe()
  }

  private emit(type: keyof DesktopMainGatewaySocket['listeners'], event: any): void {
    for (const listener of [...this.listeners[type]]) {
      listener(event)
    }
  }
}
