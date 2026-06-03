export type GatewayEventName =
  | 'gateway.ready'
  | 'session.info'
  | 'message.start'
  | 'message.delta'
  | 'message.complete'
  | 'thinking.delta'
  | 'reasoning.delta'
  | 'reasoning.available'
  | 'status.update'
  | 'tool.start'
  | 'tool.progress'
  | 'tool.complete'
  | 'tool.generating'
  | 'clarify.request'
  | 'approval.request'
  | 'sudo.request'
  | 'secret.request'
  | 'background.complete'
  | 'error'
  | 'skin.changed'
  | (string & {})

export interface GatewayEvent<P = unknown> {
  payload?: P
  session_id?: string
  type: GatewayEventName
}

export type ConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'
export type GatewayRequestId = number | string

export interface JsonRpcFrame {
  error?: { message?: string }
  id?: GatewayRequestId | null
  method?: string
  params?: GatewayEvent
  result?: unknown
}

export type WebSocketLike = WebSocket

type PendingCall = {
  reject: (error: Error) => void
  resolve: (value: unknown) => void
  timer?: ReturnType<typeof setTimeout>
}

export interface GatewayClientOptions {
  closedErrorMessage?: string
  connectErrorMessage?: string
  createRequestId?: (nextId: number) => GatewayRequestId
  requestIdPrefix?: string
  requestTimeoutMs?: number
  socketFactory?: (url: string) => WebSocketLike
  notConnectedErrorMessage?: string
}

const ANY = '*'
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000

export class JsonRpcGatewayClient {
  private nextId = 0
  private pending = new Map<GatewayRequestId, PendingCall>()
  private socket: WebSocketLike | null = null
  private state: ConnectionState = 'idle'
  private readonly eventHandlers = new Map<string, Set<(event: GatewayEvent) => void>>()
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  private readonly options: Required<Omit<GatewayClientOptions, 'socketFactory'>> &
    Pick<GatewayClientOptions, 'socketFactory'>

  constructor(options: GatewayClientOptions = {}) {
    this.options = {
      closedErrorMessage: options.closedErrorMessage ?? 'WebSocket closed',
      connectErrorMessage: options.connectErrorMessage ?? 'WebSocket connection failed',
      createRequestId:
        options.createRequestId ?? ((nextId: number) => `${options.requestIdPrefix ?? 'r'}${nextId}`),
      notConnectedErrorMessage: options.notConnectedErrorMessage ?? 'gateway not connected',
      requestIdPrefix: options.requestIdPrefix ?? 'r',
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      socketFactory: options.socketFactory
    }
  }

  get connectionState(): ConnectionState {
    return this.state
  }

  async connect(wsUrl: string): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN || this.state === 'connecting') {
      return
    }

    this.setState('connecting')

    const socket = this.options.socketFactory?.(wsUrl) ?? new WebSocket(wsUrl)
    this.socket = socket

    socket.addEventListener('message', message => {
      this.handleMessage(message.data)
    })

    socket.addEventListener('close', () => {
      this.setState('closed')
      this.rejectAllPending(new Error(this.options.closedErrorMessage))
    })

    await new Promise<void>((resolve, reject) => {
      const onOpen = () => {
        socket.removeEventListener('error', onError)
        this.setState('open')
        resolve()
      }

      const onError = () => {
        socket.removeEventListener('open', onOpen)
        this.setState('error')
        reject(new Error(this.options.connectErrorMessage))
      }

      socket.addEventListener('open', onOpen, { once: true })
      socket.addEventListener('error', onError, { once: true })
    })
  }

  close(): void {
    this.socket?.close()
    this.socket = null
  }

  on<P = unknown>(type: GatewayEventName, handler: (event: GatewayEvent<P>) => void): () => void {
    let handlers = this.eventHandlers.get(type)

    if (!handlers) {
      handlers = new Set()
      this.eventHandlers.set(type, handlers)
    }

    handlers.add(handler as (event: GatewayEvent) => void)

    return () => handlers?.delete(handler as (event: GatewayEvent) => void)
  }

  onAny(handler: (event: GatewayEvent) => void): () => void {
    return this.on(ANY as GatewayEventName, handler)
  }

  onEvent(handler: (event: GatewayEvent) => void): () => void {
    return this.onAny(handler)
  }

  onState(handler: (state: ConnectionState) => void): () => void {
    this.stateHandlers.add(handler)
    handler(this.state)

    return () => this.stateHandlers.delete(handler)
  }

  request<T>(method: string, params: Record<string, unknown> = {}, timeoutMs = this.options.requestTimeoutMs): Promise<T> {
    const socket = this.socket

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error(this.options.notConnectedErrorMessage))
    }

    const id = this.options.createRequestId(++this.nextId)

    return new Promise<T>((resolve, reject) => {
      const pending: PendingCall = {
        reject,
        resolve: value => resolve(value as T)
      }

      if (timeoutMs > 0) {
        pending.timer = setTimeout(() => {
          if (this.pending.delete(id)) {
            reject(new Error(`request timed out: ${method}`))
          }
        }, timeoutMs)
      }

      this.pending.set(id, pending)

      try {
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id,
            method,
            params
          })
        )
      } catch (error) {
        this.clearPending(id)
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  private handleMessage(raw: unknown): void {
    const text = typeof raw === 'string' ? raw : String(raw)
    let frame: JsonRpcFrame

    try {
      frame = JSON.parse(text) as JsonRpcFrame
    } catch {
      return
    }

    if (frame.id !== undefined && frame.id !== null) {
      const call = this.pending.get(frame.id)

      if (!call) {
        return
      }

      this.clearPending(frame.id)

      if (frame.error) {
        call.reject(new Error(frame.error.message || 'Hermes RPC failed'))
      } else {
        call.resolve(frame.result)
      }

      return
    }

    if (frame.method === 'event' && frame.params?.type) {
      this.dispatchEvent(frame.params)
    }
  }

  private clearPending(id: GatewayRequestId): void {
    const call = this.pending.get(id)

    if (call?.timer) {
      clearTimeout(call.timer)
    }

    this.pending.delete(id)
  }

  private dispatchEvent(event: GatewayEvent): void {
    for (const handler of this.eventHandlers.get(event.type) ?? []) {
      handler(event)
    }

    for (const handler of this.eventHandlers.get(ANY) ?? []) {
      handler(event)
    }
  }

  private rejectAllPending(error: Error): void {
    for (const [id, call] of this.pending) {
      if (call.timer) {
        clearTimeout(call.timer)
      }

      call.reject(error)
      this.pending.delete(id)
    }
  }

  private setState(state: ConnectionState): void {
    if (this.state === state) {
      return
    }

    this.state = state

    for (const handler of this.stateHandlers) {
      handler(state)
    }
  }
}
