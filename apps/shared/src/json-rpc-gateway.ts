export type GatewayEventName =
  | 'gateway.ready'
  | 'session.info'
  | 'session.usage'
  | 'message.start'
  | 'message.delta'
  | 'message.interim'
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
  /** Renderer-side source tag added by the Desktop gateway registry. */
  profile?: string
  /** Registry connection whose socket delivered the event (renderer-side tag;
   * absent for the local/legacy primary path). */
  connectionId?: string
  session_id?: string
  type: GatewayEventName
}

export type ConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'
export type GatewayRequestId = number | string

export interface JsonRpcErrorPayload {
  code?: number
  data?: unknown
  message?: string
}

export interface JsonRpcFrame {
  error?: JsonRpcErrorPayload
  id?: GatewayRequestId | null
  method?: string
  params?: GatewayEvent
  result?: unknown
}

/** JSON-RPC error with optional structured `data` from the gateway. */
export class JsonRpcGatewayError extends Error {
  readonly code?: number
  readonly data?: unknown

  constructor(message: string, options?: { code?: number; data?: unknown }) {
    super(message)
    this.name = 'JsonRpcGatewayError'
    this.code = options?.code
    this.data = options?.data
  }
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
  connectTimeoutMs?: number
  createRequestId?: (nextId: number) => GatewayRequestId
  heartbeatDeadlineMs?: number
  heartbeatIntervalMs?: number
  /** Return true to intercept the default closed-state transition. */
  onSocketClose?: (event: CloseEvent) => boolean | void
  requestIdPrefix?: string
  requestTimeoutMs?: number
  socketFactory?: (url: string) => WebSocketLike
  notConnectedErrorMessage?: string
}

const ANY = '*'
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000
// Replay fetch after reconnect: bounded so a wedged backend can't hold the
// guard open; generous enough for a 512-frame ring to drain.
const REPLAY_REQUEST_TIMEOUT_MS = 10_000
const DEFAULT_HEARTBEAT_INTERVAL_MS = 15_000
const DEFAULT_HEARTBEAT_DEADLINE_MS = 45_000
// A reconnect after sleep/wake must not hang forever in 'connecting' (which
// keeps the composer disabled and stuck on "Starting Hermes..."). If the open
// handshake doesn't land in this window, fail to 'error' so callers can retry.
const DEFAULT_CONNECT_TIMEOUT_MS = 15_000

export class JsonRpcGatewayClient {
  private nextId = 0
  private pending = new Map<GatewayRequestId, PendingCall>()
  private socket: WebSocketLike | null = null
  private state: ConnectionState = 'idle'
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private heartbeatSequence = 0
  private lastInboundAt = 0
  /** Last observed event seq per session_id — drives lossless reconnect replay. */
  private lastSeenSeq = new Map<string, number>()
  /** Set while a post-reconnect replay fetch is in flight (dedup guard). */
  private replayInFlight = false
  /** Server boot epoch from gateway.ready — seq watermarks are only valid
   * within one epoch; a change (gateway restart) resets them. */
  private serverEpoch: string | null = null
  /** Seqs dispatched LIVE while a replay RPC is in flight, per session —
   * the replay response overlaps with these and must not re-dispatch them. */
  private liveSeqsDuringReplay: Map<string, Set<number>> | null = null
  private readonly eventHandlers = new Map<string, Set<(event: GatewayEvent) => void>>()
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  private readonly options: Required<Omit<GatewayClientOptions, 'socketFactory'>> &
    Pick<GatewayClientOptions, 'socketFactory'>

  constructor(options: GatewayClientOptions = {}) {
    this.options = {
      closedErrorMessage: options.closedErrorMessage ?? 'WebSocket closed',
      connectErrorMessage: options.connectErrorMessage ?? 'WebSocket connection failed',
      connectTimeoutMs: options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS,
      createRequestId: options.createRequestId ?? ((nextId: number) => `${options.requestIdPrefix ?? 'r'}${nextId}`),
      heartbeatDeadlineMs: options.heartbeatDeadlineMs ?? DEFAULT_HEARTBEAT_DEADLINE_MS,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      notConnectedErrorMessage: options.notConnectedErrorMessage ?? 'gateway not connected',
      onSocketClose: options.onSocketClose ?? (() => false),
      requestIdPrefix: options.requestIdPrefix ?? 'r',
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      socketFactory: options.socketFactory
    }
  }

  get connectionState(): ConnectionState {
    return this.state
  }

  async connect(wsUrl: string): Promise<void> {
    // Refuse garbage; WebSocket coerces non-strings into
    // `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).
    const invalidUrl = () => {
      const got = typeof wsUrl === 'string' ? JSON.stringify(wsUrl) : `type "${typeof wsUrl}"`

      return new Error(`gateway connect() requires a ws:// or wss:// URL string, got ${got}`)
    }

    if (typeof wsUrl !== 'string') {
      throw invalidUrl()
    }

    let url: URL

    try {
      url = new URL(wsUrl)
    } catch {
      throw invalidUrl()
    }

    if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
      throw invalidUrl()
    }

    if (this.socket?.readyState === WebSocket.OPEN || this.state === 'connecting') {
      return
    }

    this.setState('connecting')

    const socket = this.options.socketFactory?.(wsUrl) ?? new WebSocket(wsUrl)
    this.socket = socket
    this.stopHeartbeat()

    socket.addEventListener('message', message => {
      if (this.socket !== socket) {
        return
      }

      this.lastInboundAt = Date.now()
      this.handleMessage(message.data)
    })

    socket.addEventListener('close', event => {
      if (this.socket !== socket) {
        return
      }

      if (this.options.onSocketClose(event)) {
        return
      }

      this.socket = null
      this.stopHeartbeat()
      this.setState('closed')
      this.rejectAllPending(new Error(this.options.closedErrorMessage))
    })

    await new Promise<void>((resolve, reject) => {
      let settled = false
      let timer: ReturnType<typeof setTimeout> | undefined

      const cleanup = () => {
        if (timer !== undefined) {
          clearTimeout(timer)
        }

        socket.removeEventListener('open', onOpen)
        socket.removeEventListener('error', onError)
      }

      const onOpen = () => {
        if (settled || this.socket !== socket) {
          return
        }

        settled = true
        cleanup()
        this.setState('open')
        resolve()
        // Lossless resume: drain events emitted while we were disconnected.
        // Fire-and-forget so connect() latency is unaffected; only runs when
        // we actually observed seq'd events before the drop.
        void this.fetchReplay()
      }

      const onError = () => {
        if (settled || this.socket !== socket) {
          return
        }

        settled = true
        cleanup()
        this.setState('error')
        reject(new Error(this.options.connectErrorMessage))
      }

      socket.addEventListener('open', onOpen, { once: true })
      socket.addEventListener('error', onError, { once: true })

      if (this.options.connectTimeoutMs > 0) {
        timer = setTimeout(() => {
          if (settled) {
            return
          }

          settled = true
          cleanup()

          // Drop the half-open socket so the next connect() starts clean
          // instead of short-circuiting on a zombie 'connecting' state.
          if (this.socket === socket) {
            try {
              socket.close()
            } catch {
              // ignore
            }

            this.socket = null
            this.setState('error')
          }

          reject(new Error(this.options.connectErrorMessage))
        }, this.options.connectTimeoutMs)
      }
    })
  }

  close(): void {
    const socket = this.socket

    if (!socket) {
      return
    }

    try {
      socket.close()
    } finally {
      this.socket = null
      this.stopHeartbeat()
      this.setState('closed')
      this.rejectAllPending(new Error(this.options.closedErrorMessage))
    }
  }

  /**
   * Invalidate the current socket generation after an ambiguous transport
   * outcome. The outer connection owner decides whether/when to reconnect.
   */
  invalidate(message = this.options.closedErrorMessage): void {
    const socket = this.socket

    if (!socket) {
      return
    }

    this.invalidateSocket(socket, new Error(message))
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

  request<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = this.options.requestTimeoutMs,
    signal?: AbortSignal
  ): Promise<T> {
    const socket = this.socket

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error(this.options.notConnectedErrorMessage))
    }

    if (signal?.aborted) {
      return Promise.reject(new DOMException('Aborted', 'AbortError'))
    }

    const id = this.options.createRequestId(++this.nextId)

    return new Promise<T>((resolve, reject) => {
      let onAbort: (() => void) | undefined

      const detach = () => {
        if (onAbort && signal) {
          signal.removeEventListener('abort', onAbort)
        }
      }

      const pending: PendingCall = {
        resolve: value => {
          detach()
          resolve(value as T)
        },
        reject: error => {
          detach()
          reject(error)
        }
      }

      if (timeoutMs > 0) {
        pending.timer = setTimeout(() => {
          if (this.pending.delete(id)) {
            detach()
            // Include the configured timeout so a caller (or a user looking
            // at an error toast) can tell whether the default 30s window
            // fired or a per-call override — e.g. /compress opts into 120s.
            const seconds = Math.round(timeoutMs / 1000)
            reject(new Error(`request timed out after ${seconds}s: ${method}`))
          }
        }, timeoutMs)
      }

      // Abort drops the pending call immediately (no dangling resolver/timer);
      // server-side cancellation is a separate cooperative RPC where it matters.
      if (signal) {
        onAbort = () => {
          const call = this.pending.get(id)

          if (call?.timer) {
            clearTimeout(call.timer)
          }

          this.pending.delete(id)
          detach()
          reject(new DOMException('Aborted', 'AbortError'))
        }

        signal.addEventListener('abort', onAbort, { once: true })
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
        detach()
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
        call.reject(
          new JsonRpcGatewayError(frame.error.message || 'Hermes RPC failed', {
            code: typeof frame.error.code === 'number' ? frame.error.code : undefined,
            data: frame.error.data
          })
        )
      } else {
        call.resolve(frame.result)
      }

      return
    }

    if (frame.method === 'event' && frame.params?.type) {
      if (frame.params.type === 'gateway.ready') {
        if (this.gatewayReadyAdvertisesHeartbeat(frame.params.payload)) {
          const socket = this.socket

          if (socket) {
            this.startHeartbeat(socket)
          }
        }

        // Seq-namespace epoch: a new epoch means the gateway restarted and
        // its per-session seq counters reset. Our stored watermarks are from
        // the previous namespace — a stale HIGH watermark would make every
        // replay return empty ("client ahead") and can suppress gap
        // detection forever. Drop all watermarks so this connection starts
        // fresh; the app layer re-hydrates state via session.resume anyway.
        const payload = frame.params.payload as { epoch?: unknown } | undefined
        const epoch = typeof payload?.epoch === 'string' ? payload.epoch : null

        if (epoch) {
          if (this.serverEpoch && this.serverEpoch !== epoch) {
            this.lastSeenSeq.clear()
          }

          this.serverEpoch = epoch
        }
      }

      this.recordSeq(frame.params)

      // While a replay RPC is in flight, remember which seqs arrived live —
      // the replay response overlaps with them (server returns everything
      // > our pre-replay watermark) and must not re-dispatch those.
      if (this.liveSeqsDuringReplay) {
        const sid = frame.params.session_id
        const seq = (frame.params as { seq?: unknown }).seq

        if (sid && typeof seq === 'number') {
          let set = this.liveSeqsDuringReplay.get(sid)

          if (!set) {
            set = new Set()
            this.liveSeqsDuringReplay.set(sid, set)
          }

          set.add(seq)
        }
      }

      this.dispatchEvent(frame.params)
    }
  }

  /**
   * Track each session's last observed event seq. Events without a seq
   * (legacy backend, session-less globals) leave the map untouched.
   */
  private recordSeq(event: GatewayEvent): void {
    const sid = event.session_id
    const seq = (event as { seq?: unknown }).seq

    if (!sid || typeof seq !== 'number' || !Number.isFinite(seq)) {
      return
    }

    const prev = this.lastSeenSeq.get(sid) ?? 0

    if (seq > prev) {
      this.lastSeenSeq.set(sid, seq)
    }
  }

  /** Test/telemetry hook: current last-seen seq map snapshot. */
  getSeqWatermarks(): Record<string, number> {
    return Object.fromEntries(this.lastSeenSeq)
  }

  /**
   * After a reconnect, ask the gateway to replay every event newer than our
   * per-session watermarks. Replayed frames go through the SAME dispatchEvent
   * path as live frames. Frames that arrived LIVE while the replay RPC was in
   * flight are tracked and skipped here — the server returns everything past
   * the pre-replay watermark, so those overlap, and re-dispatching them would
   * double-append streamed text (message.delta has no identity; it's
   * append-only downstream).
   * Best-effort: failures are swallowed (the next reconnect retries).
   */
  private async fetchReplay(): Promise<void> {
    if (this.replayInFlight || this.lastSeenSeq.size === 0) {
      return
    }

    this.replayInFlight = true
    this.liveSeqsDuringReplay = new Map()

    try {
      const entries = Object.entries(this.getSeqWatermarks())
      // One RPC per known session keeps params flat; sessions are few (<20).
      await Promise.allSettled(
        entries.map(async ([sid, lastSeen]) => {
          const result = await this.request<{
            events?: Array<{ type: string; session_id?: string; seq?: number; payload?: unknown }>
            latest_seq?: number
            truncated?: boolean
          }>('session.events.since', { session_id: sid, last_seen: lastSeen }, REPLAY_REQUEST_TIMEOUT_MS)

          // Seq epoch reset (gateway restart / server-side eviction): the
          // server's counter is now BEHIND our watermark, so replay could
          // never deliver again. Re-adopt the server's epoch so future
          // reconnects work; the gap itself is unrecoverable (truncated).
          const latest = typeof result?.latest_seq === 'number' ? result.latest_seq : 0

          if (result?.truncated && latest < lastSeen) {
            if (latest > 0) {
              this.lastSeenSeq.set(sid, latest)
            } else {
              this.lastSeenSeq.delete(sid)
            }

            return
          }

          if (!Array.isArray(result?.events)) {
            return
          }

          const liveSeqs = this.liveSeqsDuringReplay?.get(sid)

          for (const event of result.events) {
            if (!event?.type) {
              continue
            }

            // Skip events already dispatched live during the replay window.
            if (typeof event.seq === 'number' && liveSeqs?.has(event.seq)) {
              continue
            }

            this.recordSeq(event as GatewayEvent)
            this.dispatchEvent(event as GatewayEvent)
          }
        })
      )
    } catch {
      // Replay is an optimization over lossy-reconnect; never surface errors.
    } finally {
      this.replayInFlight = false
      this.liveSeqsDuringReplay = null
    }
  }

  private gatewayReadyAdvertisesHeartbeat(payload: unknown): boolean {
    return Boolean(payload && typeof payload === 'object' && (payload as { heartbeat?: unknown }).heartbeat === true)
  }

  private startHeartbeat(socket: WebSocketLike): void {
    this.stopHeartbeat()
    this.lastInboundAt = Date.now()

    if (this.options.heartbeatIntervalMs <= 0 || this.options.heartbeatDeadlineMs <= 0) {
      return
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.socket !== socket || socket.readyState !== WebSocket.OPEN) {
        return
      }

      if (Date.now() - this.lastInboundAt >= this.options.heartbeatDeadlineMs) {
        this.invalidateSocket(socket, new Error('WebSocket heartbeat acknowledgement timed out'))

        return
      }

      try {
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id: `heartbeat-${++this.heartbeatSequence}`,
            method: 'gateway.ping',
            params: {}
          })
        )
      } catch (error) {
        this.invalidateSocket(socket, error instanceof Error ? error : new Error(String(error)))
      }
    }, this.options.heartbeatIntervalMs)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private invalidateSocket(socket: WebSocketLike, error: Error): void {
    if (this.socket !== socket) {
      return
    }

    this.socket = null
    this.stopHeartbeat()

    try {
      socket.close()
    } catch {
      // The generation was already invalidated; the reconnect owner can redial.
    }

    this.setState('closed')
    this.rejectAllPending(error)
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
