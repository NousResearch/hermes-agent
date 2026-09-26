import type { GatewayEvent, GatewayEventName } from './gateway-events.js'
import {
  DEFAULT_HEARTBEAT_DEADLINE_MS,
  DEFAULT_HEARTBEAT_INTERVAL_MS,
  type GatewayRequestId,
  JsonRpcRequestChannel,
  type JsonRpcRequestChannelOptions,
  type JsonRpcTransport,
  type ServerRequestHandler,
  wireFrameText
} from './json-rpc-channel.js'

export type { GatewayEvent, GatewayEventName } from './gateway-events.js'
export type ConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

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
  /**
   * When `connect()` settles. `'gateway.ready'` (default) waits for that first
   * frame; any other first frame is a protocol failure. `'open'` settles on
   * the WebSocket `open` event — for sockets such as `/api/events` that never
   * send `gateway.ready`.
   */
  handshake?: 'gateway.ready' | 'open'
  createRequestId?: (nextId: number) => GatewayRequestId
  heartbeatDeadlineMs?: number
  heartbeatIntervalMs?: number
  /** A server→client request handler threw; the channel already answered `-32603`. */
  onRequestHandlerError?: JsonRpcRequestChannelOptions['onRequestHandlerError']
  /** No handler accepted a server→client request; the channel already answered `-32601`. */
  onUnhandledRequest?: JsonRpcRequestChannelOptions['onUnhandledRequest']
  /** Return true to intercept the default closed-state transition. */
  onSocketClose?: (event: { code: number }) => boolean | void
  /** Fetch `session.events.since` after a reconnect (default). Off for notification-only feeds whose peer never answers RPCs. */
  replay?: boolean
  requestIdPrefix?: string
  requestTimeoutMs?: number
  socketFactory?: (url: string) => WebSocketLike
  notConnectedErrorMessage?: string
}

const ANY = '*'
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000

const isGatewayReady = (event: GatewayEvent): event is GatewayEvent<'gateway.ready'> => event.type === 'gateway.ready'
// Replay fetch after reconnect: bounded so a wedged backend can't hold the
// guard open; generous enough for a 512-frame ring to drain.
const REPLAY_REQUEST_TIMEOUT_MS = 10_000
// A reconnect after sleep/wake must not hang forever in 'connecting' (which
// keeps the composer disabled and stuck on "Starting Hermes..."). If the open
// handshake doesn't land in this window, fail to 'error' so callers can retry.
const DEFAULT_CONNECT_TIMEOUT_MS = 15_000

/** True for a `ws://` / `wss://` URL string — the only thing `JsonRpcGatewayClient.connect()` will dial. */
export function isGatewayWebSocketUrl(value: unknown): value is string {
  if (typeof value !== 'string') {
    return false
  }

  try {
    const protocol = new URL(value).protocol

    return protocol === 'ws:' || protocol === 'wss:'
  } catch {
    return false
  }
}

/**
 * Typed fan-out of gateway `event` notifications: per-type handlers plus a
 * `*` wildcard. Shared by the WebSocket client below and the Ink TUI's stdio
 * client so both dispatch the same way.
 */
export class GatewayEventHub {
  private readonly handlers = new Map<string, Set<(event: GatewayEvent) => void>>()

  on<K extends GatewayEventName>(type: K, handler: (event: GatewayEvent<K>) => void): () => void {
    let set = this.handlers.get(type)

    if (!set) {
      set = new Set()
      this.handlers.set(type, set)
    }

    set.add(handler as (event: GatewayEvent) => void)

    return () => set?.delete(handler as (event: GatewayEvent) => void)
  }

  onAny(handler: (event: GatewayEvent) => void): () => void {
    // ANY is a client-side wildcard, not a wire name; it never reaches the typed map.
    return this.on(ANY as GatewayEventName, handler as (event: GatewayEvent<GatewayEventName>) => void)
  }

  dispatch(event: GatewayEvent): void {
    for (const handler of this.handlers.get(event.type) ?? []) {
      handler(event)
    }

    for (const handler of this.handlers.get(ANY) ?? []) {
      handler(event)
    }
  }
}

/**
 * Bring a `JsonRpcRequestChannel` to a raw text sink — a WebSocket here, a
 * child's stdin in the TUI. Kept separate from the socket so the channel never
 * holds a reference to a specific socket generation.
 */
const socketTransport = (socket: WebSocketLike): JsonRpcTransport => ({ send: text => socket.send(text) })

interface SessionReplay {
  events: GatewayEvent[]
  promise: Promise<boolean>
  resolve: (valid: boolean) => void
}

interface ReplayPlan {
  entries: Array<[string, number]>
  generation: number
}

export class JsonRpcGatewayClient {
  private socket: WebSocketLike | null = null
  private state: ConnectionState = 'idle'
  private readonly channel: JsonRpcRequestChannel
  private readonly events = new GatewayEventHub()
  /** Last observed event seq per session_id — drives lossless reconnect replay. */
  private lastSeenSeq = new Map<string, number>()
  /** Invalidates an interrupted replay so its async cleanup cannot own a replacement socket. */
  private replayGeneration = 0
  /**
   * While a replay fetch is in flight, live seq'd frames for the sessions
   * being replayed are parked here instead of dispatching immediately.
   * Without this hold, a live frame racing the replay response is dispatched
   * twice (once live, once when the replay returns the same seq) or, worse,
   * advances the watermark so the gap events the replay carries get skipped.
   */
  private replayHold: Map<string, SessionReplay> | null = null
  /**
   * Server process identity for the replay contract (from gateway.ready /
   * session.events.since). Seq counters are in-process on the backend, so a
   * restart resets them while we still hold high watermarks — without this
   * check events_since(sid, 97) returns [] + truncated=false forever and we
   * silently believe nothing was missed.
   */
  private replayEpoch: string | null = null
  private attempt: ConnectAttempt | null = null
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  private readonly options: Required<
    Omit<GatewayClientOptions, 'onRequestHandlerError' | 'onUnhandledRequest' | 'socketFactory'>
  > &
    Pick<GatewayClientOptions, 'onRequestHandlerError' | 'onUnhandledRequest' | 'socketFactory'>

  constructor(options: GatewayClientOptions = {}) {
    const connectErrorMessage = options.connectErrorMessage ?? 'WebSocket connection failed'

    this.options = {
      authRejectedErrorMessage: options.authRejectedErrorMessage ?? connectErrorMessage,
      closedErrorMessage: options.closedErrorMessage ?? 'WebSocket closed',
      connectErrorMessage,
      connectTimeoutMs: options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS,
      handshake: options.handshake ?? 'gateway.ready',
      createRequestId: options.createRequestId ?? ((nextId: number) => `${options.requestIdPrefix ?? 'r'}${nextId}`),
      heartbeatDeadlineMs: options.heartbeatDeadlineMs ?? DEFAULT_HEARTBEAT_DEADLINE_MS,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      notConnectedErrorMessage: options.notConnectedErrorMessage ?? 'gateway not connected',
      onSocketClose: options.onSocketClose ?? (() => false),
      replay: options.replay ?? true,
      requestIdPrefix: options.requestIdPrefix ?? 'r',
      onRequestHandlerError: options.onRequestHandlerError,
      onUnhandledRequest: options.onUnhandledRequest,
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      socketFactory: options.socketFactory
    }
    this.channel = new JsonRpcRequestChannel({
      createRequestId: this.options.createRequestId,
      heartbeatDeadlineMs: this.options.heartbeatDeadlineMs,
      heartbeatIntervalMs: this.options.heartbeatIntervalMs,
      // Desktop/web and the TUI alike count any inbound frame as liveness
      // (#115251): streamed deltas are life; only a silent drop trips the
      // deadline.
      heartbeatLiveness: 'any-inbound',
      onEvent: event => this.handleEvent(event),
      onHeartbeatFailure: error => this.invalidate(error.message),
      onRequestHandlerError: this.options.onRequestHandlerError,
      onUnhandledRequest: this.options.onUnhandledRequest,
      requestTimeoutMs: this.options.requestTimeoutMs
    })
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

    if (!isGatewayWebSocketUrl(wsUrl)) {
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

      return Promise.reject(this.connectFailure('WebSocket error before open'))
    }

    const transport = socketTransport(socket)
    this.socket = socket
    this.channel.stopHeartbeat()

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

    const ownsTransport = () => this.socket === socket && this.channel.owns(transport)

    const onOpen = () => {
      if (this.socket !== socket || this.attempt !== attempt || attempt.settled) {
        return
      }

      // A raw WebSocket open is only transport readiness. The connection stays
      // in 'connecting' until the gateway identifies itself with gateway.ready.
      if (this.options.handshake !== 'open') {
        return
      }

      if (!this.settleConnectAttempt(attempt)) {
        return
      }

      this.channel.attach(transport)
      const plan = this.installReplayBarriers()
      this.setState('open')

      if (!ownsTransport()) {
        attempt.reject(new GatewayConnectError(this.options.closedErrorMessage))

        return
      }

      this.issueReplay(plan)
      attempt.resolve()
    }

    const onError = () => {
      if (this.socket !== socket || this.attempt !== attempt || attempt.settled) {
        return
      }

      if (!this.settleConnectAttempt(attempt)) {
        return
      }

      this.setState('error')
      attempt.reject(this.connectFailure('WebSocket error before open'))
    }

    socket.addEventListener('message', message => {
      if (this.socket !== socket) {
        return
      }

      const parsed = this.parseMessage(message.data)

      if (this.attempt === attempt && !attempt.settled && this.options.handshake === 'gateway.ready') {
        const frame = parsed?.frame

        if (parsed && frame?.method === 'event' && frame.params?.type === 'gateway.ready') {
          if (!this.settleConnectAttempt(attempt)) {
            return
          }

          this.channel.attach(transport)
          const plan = this.installReplayBarriers()
          this.setState('open')

          if (!ownsTransport()) {
            attempt.reject(new GatewayConnectError(this.options.closedErrorMessage))

            return
          }

          this.channel.handleFrame(parsed.text)

          if (!ownsTransport()) {
            attempt.reject(new GatewayConnectError(this.options.closedErrorMessage))

            return
          }

          this.issueReplay(plan)
          attempt.resolve()

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

      if (parsed) {
        this.channel.handleFrame(parsed.text)
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
          this.connectFailure(
            `WebSocket closed during handshake: code ${event.code}${event.reason ? ` ${event.reason}` : ''}`,
            {
              wsCloseCode: event.code,
              needsOauthLogin: needsOauthLogin || undefined
            },
            needsOauthLogin ? this.options.authRejectedErrorMessage : this.options.connectErrorMessage
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

        this.dropSocket(new Error(this.options.closedErrorMessage))

        return
      }

      // A failed handshake may close after its error/protocol-failure path has
      // already settled. Release that socket without overwriting 'error'.
      this.socket = null
      this.channel.stopHeartbeat()
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
        attempt.reject(this.connectFailure(`no WebSocket open within ${this.options.connectTimeoutMs} ms`))
      }, this.options.connectTimeoutMs)
    }

    return promise
  }

  private connectFailure(
    detail: string,
    options?: { wsCloseCode?: number; needsOauthLogin?: boolean },
    message = this.options.connectErrorMessage
  ): GatewayConnectError {
    return new GatewayConnectError(`${message} (${detail})`, options)
  }

  close(): void {
    const attempt = this.attempt

    // Settle a pending attempt eagerly before invalidate() drops the socket.
    // A raw-open socket was a real transport waiting on gateway.ready; a
    // CONNECTING socket never established one.
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

    this.invalidate()
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

    // Drop the generation BEFORE closing: a synchronous `close` event from
    // the socket must hit the identity guard and not run the default
    // closed-path a second time on top of whatever the owner redialed.
    this.dropSocket(new Error(message))

    try {
      socket.close()
    } catch {
      // The generation was already invalidated; the reconnect owner can redial.
    }
  }

  on<K extends GatewayEventName>(type: K, handler: (event: GatewayEvent<K>) => void): () => void {
    return this.events.on(type, handler)
  }

  onAny(handler: (event: GatewayEvent) => void): () => void {
    return this.events.onAny(handler)
  }

  onEvent(handler: (event: GatewayEvent) => void): () => void {
    return this.onAny(handler)
  }

  /**
   * Server→client requests (clarify, approval, sudo, …). Live frames and
   * `open_requests` re-delivered after a reconnect both arrive here; the
   * latter carry `replayed: true`.
   */
  onRequest(handler: ServerRequestHandler): () => void {
    return this.channel.onRequest(handler)
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

    return this.channel.request<T>(
      method,
      params,
      timeoutMs,
      signal,
      () => new Error(this.options.notConnectedErrorMessage)
    )
  }

  private parseMessage(raw: unknown): {
    frame: { method?: unknown; params?: GatewayEvent }
    text: string
  } | null {
    const text = wireFrameText(raw)

    if (text === null) {
      return null
    }

    let parsed: unknown

    try {
      parsed = JSON.parse(text)
    } catch {
      return null
    }

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return null
    }

    return {
      frame: parsed as { method?: unknown; params?: GatewayEvent },
      text
    }
  }

  private handleEvent(event: GatewayEvent): void {
    if (isGatewayReady(event)) {
      if (event.payload?.heartbeat === true) {
        this.channel.startHeartbeat()
      }

      const epoch = event.payload?.replay_epoch

      if (typeof epoch === 'string' && epoch) {
        this.adoptReplayEpoch(epoch)
      }
    }

    const sid = event.session_id
    const seqValue = event.seq

    if (this.replayHold && sid && typeof seqValue === 'number' && this.replayHold.has(sid)) {
      // Replay in flight for this session: park the frame; flushReplayHold
      // dispatches it after the replayed gap, gated on seq.
      this.replayHold.get(sid)?.events.push(event)

      return
    }

    this.recordSeq(event)
    this.dispatchEvent(event)
  }

  /**
   * Track each session's last observed event seq. Events without a seq
   * (legacy backend, session-less globals) leave the map untouched.
   */
  private recordSeq(event: GatewayEvent): void {
    const sid = event.session_id
    const seq = event.seq

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
   * Wait for this session's reconnect replay AND parked live frames to dispatch.
   * True includes bounded timeout/unsupported-method fallback and an epoch
   * change on the still-open socket (backend restart: nothing will replay, so
   * REST is authoritative); false means the socket was lost and a pending
   * history read must be abandoned (the next open re-reads).
   * Unobserved sessions and replay-disabled feeds have no barrier.
   */
  sessionReplayBarrier(sessionId: string): Promise<boolean> | undefined {
    const pending = this.replayHold?.get(sessionId)?.promise

    if (pending) {
      return pending
    }

    // A history response can beat the replacement connection itself. Don't
    // publish ahead of a replay that will only be installed on the next open.
    if (this.options.replay && this.lastSeenSeq.has(sessionId) && this.state !== 'open') {
      return Promise.resolve(false)
    }

    return undefined
  }

  /**
   * Snapshot every observed session and install its hold before open listeners
   * can read history. This phase sends no replay requests.
   */
  private installReplayBarriers(): ReplayPlan | null {
    if (!this.options.replay || this.replayHold || this.lastSeenSeq.size === 0) {
      return null
    }

    const generation = ++this.replayGeneration
    // Park live frames for the sessions we're about to replay so a frame
    // racing the replay response can't dispatch ahead of (or duplicate) the
    // gap events. Sessions without watermarks are unaffected.
    const entries = [...this.lastSeenSeq]
    const hold = new Map<string, SessionReplay>()

    for (const [sid] of entries) {
      let resolve!: (valid: boolean) => void

      const promise = new Promise<boolean>(settle => {
        resolve = settle
      })

      hold.set(sid, { events: [], promise, resolve })
    }

    this.replayHold = hold

    return { entries, generation }
  }

  /** Request each session's replay through the public, open-state-gated request path. */
  private issueReplay(plan: ReplayPlan | null): void {
    if (!plan) {
      return
    }

    // A hung background session must not hold a ready session's transcript.
    for (const [sid, lastSeen] of plan.entries) {
      void this.fetchSessionReplay(sid, lastSeen, plan.generation)
    }
  }

  private async fetchSessionReplay(sid: string, lastSeen: number, replayGeneration: number): Promise<void> {
    if (this.replayGeneration !== replayGeneration) {
      return
    }

    try {
      // `open_requests` on the answer are re-delivered by the channel itself.
      const result = await this.request<{ events?: GatewayEvent[]; epoch?: string }>(
        'session.events.since',
        { session_id: sid, last_seen: lastSeen },
        REPLAY_REQUEST_TIMEOUT_MS
      )

      // The socket that owned this replay was dropped while its requests were
      // settling. Its results and cleanup must not consume the replacement
      // socket's replay window.
      if (this.replayGeneration !== replayGeneration) {
        return
      }

      const epoch = result?.epoch

      if (typeof epoch === 'string' && epoch && this.replayEpoch && epoch !== this.replayEpoch) {
        // The old cursor no longer describes this process's numbering.
        this.adoptReplayEpoch(epoch)

        return
      }

      if (typeof epoch === 'string' && epoch && !this.replayEpoch) {
        this.replayEpoch = epoch
      }

      if (!Array.isArray(result?.events)) {
        return
      }

      for (const event of result.events) {
        // Event handlers can synchronously invalidate and replace the socket.
        if (this.replayGeneration !== replayGeneration) {
          return
        }

        if (event?.type) {
          this.dispatchIfNewer({ ...event, replayed: true })
        }
      }
    } catch {
      // Replay is an optimization over lossy-reconnect; never surface errors.
    } finally {
      if (this.replayGeneration === replayGeneration) {
        this.flushReplayHold(sid, replayGeneration)
      }
    }
  }

  /**
   * Dispatch an event only when its seq advances the session watermark.
   * Seq-less events always dispatch (no ordering contract to violate).
   */
  private dispatchIfNewer(event: GatewayEvent): void {
    const sid = event.session_id
    const seq = event.seq

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

    const changed = this.replayEpoch !== null
    this.replayEpoch = epoch

    if (changed) {
      this.lastSeenSeq.clear()
      // Revoke requests/cursors from the old numbering, but retain live
      // frames already received on this still-open socket. The socket is
      // still ours and no replay can cover the old numbering, so waiting
      // history reads proceed: REST is the only recovery left (#94779).
      // Their continuations run after the parked frames below dispatch.
      const hold = this.cancelReplay(true)
      const generation = this.replayGeneration

      for (const replay of hold?.values() ?? []) {
        for (const event of replay.events) {
          if (this.replayGeneration !== generation) {
            return
          }

          this.dispatchIfNewer({ ...event, replayed: true })
        }
      }
    }
  }

  /** Release frames parked during a replay fetch, seq-gated against dupes. */
  private flushReplayHold(sid: string, generation: number): void {
    const replay = this.replayHold?.get(sid)

    if (!replay) {
      return
    }

    // Keep the barrier visible through dispatch, including synchronous live
    // frames emitted by a handler. Remove consumed frames before callbacks
    // can revoke this epoch and flush the remainder.
    while (this.replayGeneration === generation && replay.events.length) {
      this.dispatchIfNewer({ ...replay.events.shift()!, replayed: true })
    }

    if (this.replayGeneration !== generation) {
      return
    }

    this.replayHold?.delete(sid)

    if (this.replayHold?.size === 0) {
      this.replayHold = null
    }

    replay.resolve(true)
  }

  private cancelReplay(readsMayProceed: boolean): Map<string, SessionReplay> | null {
    const hold = this.replayHold
    this.replayGeneration += 1
    this.replayHold = null

    for (const replay of hold?.values() ?? []) {
      replay.resolve(readsMayProceed)
    }

    return hold
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

  /** Forget the current socket generation, fail its calls, and go 'closed'. */
  private dropSocket(error: Error): void {
    const attempt = this.attempt

    if (attempt && !attempt.settled && this.settleConnectAttempt(attempt)) {
      attempt.reject(new GatewayConnectError(error.message))
    }

    // A replay belongs to the socket that started it. Detaching that socket
    // rejects its requests asynchronously, so clear its ownership now; the
    // next open can immediately schedule a replay of its own.
    this.cancelReplay(false)
    this.socket = null
    this.channel.detach(error)
    this.setState('closed')
  }

  private dispatchEvent(event: GatewayEvent): void {
    // Tag the frame with the process epoch this socket adopted so a consumer
    // holding several sockets to one backend can recognise the same event
    // arriving on each of them; the epoch is per process, not per socket.
    this.events.dispatch(this.replayEpoch ? { ...event, replayEpoch: this.replayEpoch } : event)
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
