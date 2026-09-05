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
  | 'todo.updated'
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

/** Connection-handshake failure with optional WebSocket close metadata. */
export class GatewayConnectError extends Error {
  readonly wsCloseCode?: number
  readonly needsOauthLogin?: boolean

  constructor(message: string, options?: { wsCloseCode?: number; needsOauthLogin?: boolean }) {
    super(message)
    this.name = 'GatewayConnectError'
    this.wsCloseCode = options?.wsCloseCode
    this.needsOauthLogin = options?.needsOauthLogin
  }
}

export type WebSocketLike = WebSocket

type PendingCall = {
  reject: (error: Error) => void
  resolve: (value: unknown) => void
  timer?: ReturnType<typeof setTimeout>
}

type ConnectAttempt = {
  socket: WebSocketLike
  url: string
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
  timer?: ReturnType<typeof setTimeout>
  settled: boolean
}

export interface GatewayClientOptions {
  authRejectedErrorMessage?: string
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
  /**
   * While a replay fetch is in flight, live seq'd frames for the sessions
   * being replayed are parked here instead of dispatching immediately.
   * Without this hold, a live frame racing the replay response is dispatched
   * twice (once live, once when the replay returns the same seq) or, worse,
   * advances the watermark so the gap events the replay carries get skipped.
   */
  private replayHold: Map<string, GatewayEvent[]> | null = null
  /**
   * Server process identity for the replay contract (from gateway.ready /
   * session.events.since). Seq counters are in-process on the backend, so a
   * restart resets them while we still hold high watermarks — without this
   * check events_since(sid, 97) returns [] + truncated=false forever and we
   * silently believe nothing was missed.
   */
  private replayEpoch: string | null = null
  private attempt: ConnectAttempt | null = null
  private readonly eventHandlers = new Map<string, Set<(event: GatewayEvent) => void>>()
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  private readonly options: Required<Omit<GatewayClientOptions, 'socketFactory'>> &
    Pick<GatewayClientOptions, 'socketFactory'>

  constructor(options: GatewayClientOptions = {}) {
    const connectErrorMessage = options.connectErrorMessage ?? 'WebSocket connection failed'

    this.options = {
      authRejectedErrorMessage: options.authRejectedErrorMessage ?? connectErrorMessage,
      closedErrorMessage: options.closedErrorMessage ?? 'WebSocket closed',
      connectErrorMessage,
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

  connect(wsUrl: string): Promise<void> {
    // Refuse garbage; WebSocket coerces non-strings into
    // `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).
    const invalidUrl = () => {
      const got = typeof wsUrl === 'string' ? JSON.stringify(wsUrl) : `type "${typeof wsUrl}"`

      return new Error(`gateway connect() requires a ws:// or wss:// URL string, got ${got}`)
    }

    if (typeof wsUrl !== 'string') {
      return Promise.reject(invalidUrl())
    }

    let url: URL

    try {
      url = new URL(wsUrl)
    } catch {
      return Promise.reject(invalidUrl())
    }

    if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
      return Promise.reject(invalidUrl())
    }

    if (this.state === 'open' && this.socket?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    if (this.attempt && !this.attempt.settled) {
      if (this.attempt.url === wsUrl) {
        return this.attempt.promise
      }

      return Promise.reject(new Error('gateway connect() already in progress'))
    }

    this.setState('connecting')

    let socket: WebSocketLike

    try {
      socket = this.options.socketFactory?.(wsUrl) ?? new WebSocket(wsUrl)
    } catch {
      this.setState('error')

      return Promise.reject(new GatewayConnectError(this.options.connectErrorMessage))
    }

    this.socket = socket
    this.stopHeartbeat()

    let resolveAttempt!: () => void
    let rejectAttempt!: (error: Error) => void

    const promise = new Promise<void>((resolve, reject) => {
      resolveAttempt = resolve
      rejectAttempt = reject
    })

    const attempt: ConnectAttempt = {
      socket,
      url: wsUrl,
      promise,
      resolve: resolveAttempt,
      reject: rejectAttempt,
      settled: false
    }

    this.attempt = attempt

    const onOpen = () => {
      if (this.socket !== socket || this.attempt !== attempt || attempt.settled) {
        return
      }

      // A raw WebSocket open is only transport readiness. The connection stays
      // in 'connecting' until the gateway identifies itself with gateway.ready.
    }

    const onError = () => {
      if (this.socket !== socket || this.attempt !== attempt || attempt.settled) {
        return
      }

      if (!this.settleConnectAttempt(attempt)) {
        return
      }

      this.setState('error')
      attempt.reject(new GatewayConnectError(this.options.connectErrorMessage))
    }

    socket.addEventListener('message', message => {
      if (this.socket !== socket) {
        return
      }

      this.lastInboundAt = Date.now()

      const frame = this.parseMessage(message.data)

      if (this.attempt === attempt && !attempt.settled) {
        if (frame?.method === 'event' && frame.params?.type === 'gateway.ready') {
          if (!this.settleConnectAttempt(attempt)) {
            return
          }

          this.setState('open')

          if (this.gatewayReadyAdvertisesHeartbeat(frame.params.payload)) {
            this.startHeartbeat(socket)
          }

          const epoch = (frame.params.payload as { replay_epoch?: unknown } | undefined)?.replay_epoch

          if (typeof epoch === 'string' && epoch) {
            this.adoptReplayEpoch(epoch)
          }

          this.dispatchEvent(frame.params)
          attempt.resolve()

          // Lossless resume: drain events emitted while we were disconnected.
          // Fire-and-forget so connect() latency is unaffected; only runs when
          // we actually observed seq'd events before the drop.
          void this.fetchReplay()

          return
        }

        if (!this.settleConnectAttempt(attempt)) {
          return
        }

        this.setState('error')

        try {
          socket.close()
        } catch {
          // ignore
        } finally {
          if (this.socket === socket) {
            this.socket = null
          }
        }

        attempt.reject(new GatewayConnectError(this.options.connectErrorMessage))

        return
      }

      if (frame) {
        this.handleFrame(frame)
      }
    })

    socket.addEventListener('close', event => {
      if (this.socket !== socket) {
        return
      }

      if (this.attempt === attempt && !attempt.settled) {
        if (!this.settleConnectAttempt(attempt)) {
          return
        }

        this.socket = null
        this.setState('closed')

        const needsOauthLogin = event.code === 4401
        attempt.reject(
          new GatewayConnectError(
            needsOauthLogin ? this.options.authRejectedErrorMessage : this.options.connectErrorMessage,
            {
              wsCloseCode: event.code,
              needsOauthLogin: needsOauthLogin || undefined
            }
          )
        )

        return
      }

      // onSocketClose is an established-connection interception hook. Handshake
      // closes are classified above and never flow through it.
      if (this.state === 'open') {
        if (this.options.onSocketClose(event)) {
          return
        }

        this.socket = null
        this.setState('closed')
        this.rejectAllPending(new Error(this.options.closedErrorMessage))

        return
      }

      // A failed handshake may close after its error/protocol-failure path has
      // already settled. Release that socket without overwriting 'error'.
      this.socket = null
      this.stopHeartbeat()
    })

    socket.addEventListener('open', onOpen, { once: true })
    socket.addEventListener('error', onError, { once: true })

    if (this.options.connectTimeoutMs > 0) {
      attempt.timer = setTimeout(() => {
        if (this.socket !== socket || this.attempt !== attempt || attempt.settled) {
          return
        }

        if (!this.settleConnectAttempt(attempt)) {
          return
        }

        // Drop the half-open socket so the next connect() starts clean
        // instead of short-circuiting on a zombie 'connecting' state.
        try {
          socket.close()
        } catch {
          // ignore
        } finally {
          if (this.socket === socket) {
            this.socket = null
            this.setState('error')
          }
        }

        this.setState('error')
        attempt.reject(new GatewayConnectError(this.options.connectErrorMessage))
      }, this.options.connectTimeoutMs)
    }

    return promise
  }

  close(): void {
    const attempt = this.attempt

    // Settle a pending attempt eagerly in both cases — close() nulls the
    // socket below, after which the connect close-listener and timeout both
    // bail on `this.socket !== socket`, so nothing else would ever settle it
    // (real sockets queue their 'close' event; only the test fakes emit it
    // synchronously). Only the message depends on how far the handshake got:
    // a socket at raw 'open' (transport established, still waiting on
    // gateway.ready) is a real close; a still-CONNECTING socket never had a
    // live transport, and per the WebSocket spec closing it fails the
    // connection rather than closing it, so it reports connectErrorMessage.
    if (attempt && !attempt.settled && this.settleConnectAttempt(attempt)) {
      this.setState('closed')
      attempt.reject(
        new GatewayConnectError(
          this.socket?.readyState === WebSocket.OPEN
            ? this.options.closedErrorMessage
            : this.options.connectErrorMessage
        )
      )
    }

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

    if (!socket || this.state !== 'open' || socket.readyState !== WebSocket.OPEN) {
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

  private parseMessage(raw: unknown): JsonRpcFrame | null {
    const text = typeof raw === 'string' ? raw : String(raw)
    let parsed: unknown

    try {
      parsed = JSON.parse(text)
    } catch {
      return null
    }

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return null
    }

    return parsed as JsonRpcFrame
  }

  private handleFrame(frame: JsonRpcFrame): void {
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

        const epoch = (frame.params.payload as { replay_epoch?: unknown } | undefined)?.replay_epoch

        if (typeof epoch === 'string' && epoch) {
          this.adoptReplayEpoch(epoch)
        }
      }

      const sid = frame.params.session_id
      const seqValue = (frame.params as { seq?: unknown }).seq

      if (this.replayHold && sid && typeof seqValue === 'number' && this.replayHold.has(sid)) {
        // Replay in flight for this session: park the frame; flushReplayHold
        // dispatches it after the replayed gap, gated on seq.
        this.replayHold.get(sid)?.push(frame.params)

        return
      }

      this.recordSeq(frame.params)
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
   * path as live frames — dedupe happens naturally because recordSeq ignores
   * non-increasing seqs and downstream stores key on event identity.
   * Best-effort: failures are swallowed (the next reconnect retries).
   */
  private async fetchReplay(): Promise<void> {
    if (this.replayInFlight || this.lastSeenSeq.size === 0) {
      return
    }

    this.replayInFlight = true
    // Park live frames for the sessions we're about to replay so a frame
    // racing the replay response can't dispatch ahead of (or duplicate) the
    // gap events. Sessions without watermarks are unaffected.
    const hold = new Map<string, GatewayEvent[]>()

    for (const sid of this.lastSeenSeq.keys()) {
      hold.set(sid, [])
    }

    this.replayHold = hold

    try {
      const entries = Object.entries(this.getSeqWatermarks())

      // One RPC per known session keeps params flat; sessions are few (<20).
      const results = await Promise.allSettled(
        entries.map(([sid, lastSeen]) =>
          this.request<{ events?: Array<{ type: string; session_id?: string; seq?: number; payload?: unknown }> }>(
            'session.events.since',
            { session_id: sid, last_seen: lastSeen },
            REPLAY_REQUEST_TIMEOUT_MS
          )
        )
      )

      for (const result of results) {
        if (result.status !== 'fulfilled' || !Array.isArray(result.value?.events)) {
          continue
        }

        const epoch = (result.value as { epoch?: unknown }).epoch

        if (typeof epoch === 'string' && epoch && this.replayEpoch && epoch !== this.replayEpoch) {
          // Backend restarted: its seq numbering reset, so our watermarks —
          // and this replay window — are meaningless. Drop them and start
          // fresh under the new epoch.
          this.adoptReplayEpoch(epoch)

          continue
        }

        if (typeof epoch === 'string' && epoch && !this.replayEpoch) {
          this.replayEpoch = epoch
        }

        for (const event of result.value.events) {
          if (!event?.type) {
            continue
          }

          this.dispatchIfNewer(event as GatewayEvent)
        }
      }
    } catch {
      // Replay is an optimization over lossy-reconnect; never surface errors.
    } finally {
      this.flushReplayHold()
      this.replayInFlight = false
    }
  }

  /**
   * Dispatch an event only when its seq advances the session watermark.
   * Seq-less events always dispatch (no ordering contract to violate).
   */
  private dispatchIfNewer(event: GatewayEvent): void {
    const sid = event.session_id
    const seq = (event as { seq?: unknown }).seq

    if (sid && typeof seq === 'number' && Number.isFinite(seq)) {
      const prev = this.lastSeenSeq.get(sid) ?? 0

      if (seq <= prev) {
        return
      }

      this.lastSeenSeq.set(sid, seq)
    }

    this.dispatchEvent(event)
  }

  /**
   * Record the server's replay epoch; on change (backend restart) the old
   * seq watermarks describe a numbering that no longer exists — clear them
   * so the next reconnect doesn't silently believe it missed nothing.
   */
  private adoptReplayEpoch(epoch: string): void {
    if (this.replayEpoch === epoch) {
      return
    }

    if (this.replayEpoch !== null) {
      this.lastSeenSeq.clear()
    }

    this.replayEpoch = epoch
  }

  /** Release frames parked during a replay fetch, seq-gated against dupes. */
  private flushReplayHold(): void {
    const hold = this.replayHold
    this.replayHold = null

    if (!hold) {
      return
    }

    for (const parked of hold.values()) {
      for (const event of parked) {
        this.dispatchIfNewer(event)
      }
    }
  }

  private settleConnectAttempt(attempt: ConnectAttempt): boolean {
    if (attempt.settled || this.attempt !== attempt) {
      return false
    }

    attempt.settled = true

    if (attempt.timer !== undefined) {
      clearTimeout(attempt.timer)
      attempt.timer = undefined
    }

    this.attempt = null

    return true
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
