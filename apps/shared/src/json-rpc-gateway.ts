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

export type WebSocketLike = WebSocket

export interface GatewayClientOptions {
  closedErrorMessage?: string
  connectErrorMessage?: string
  connectTimeoutMs?: number
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
// `approval.respond` rides the SAME deadline the backend grants the user to
// answer: tools/approval_context.py reads `approvals.timeout` (default 300s —
// gateway push notifications may not be seen for minutes). A shorter client
// timeout races that window: answer at t=130s and the frontend has already
// rejected its own RPC while the backend happily applies the decision — the
// desktop shows "request timed out" and freezes on a card that is actually
// resolved (#60654). Match the backend default so the client only gives up
// when the backend itself fails the approval closed.
export const APPROVAL_RESPOND_TIMEOUT_MS = 300_000

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
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  private readonly options: Required<
    Omit<GatewayClientOptions, 'onRequestHandlerError' | 'onUnhandledRequest' | 'socketFactory'>
  > &
    Pick<GatewayClientOptions, 'onRequestHandlerError' | 'onUnhandledRequest' | 'socketFactory'>

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

  async connect(wsUrl: string): Promise<void> {
    // Refuse garbage; WebSocket coerces non-strings into
    // `ws://<origin>/[object%20Object]` (#68250 stale-emit boot loop).
    const invalidUrl = () => {
      const got = typeof wsUrl === 'string' ? JSON.stringify(wsUrl) : `type "${typeof wsUrl}"`

      return new Error(`gateway connect() requires a ws:// or wss:// URL string, got ${got}`)
    }

    if (!isGatewayWebSocketUrl(wsUrl)) {
      throw invalidUrl()
    }

    if ((this.socket && this.socket.readyState === WebSocket.OPEN) || this.state === 'connecting') {
      return
    }

    this.setState('connecting')

    const socket = this.options.socketFactory?.(wsUrl) ?? new WebSocket(wsUrl)
    const transport = socketTransport(socket)
    this.socket = socket
    this.channel.stopHeartbeat()

    socket.addEventListener('message', message => {
      if (this.socket !== socket) {
        return
      }

      const text = wireFrameText(message.data)

      if (text !== null) {
        this.channel.handleFrame(text)
      }
    })

    socket.addEventListener('close', event => {
      if (this.socket !== socket) {
        return
      }

      if (this.options.onSocketClose(event)) {
        return
      }

      this.dropSocket(new Error(this.options.closedErrorMessage))
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
        socket.removeEventListener('close', onClose)
      }

      const onOpen = () => {
        if (settled || this.socket !== socket) {
          return
        }

        settled = true
        cleanup()
        this.channel.attach(transport)
        // Install session barriers before open listeners can start history
        // reads. Replay stays fire-and-forget; connect latency is unchanged.
        this.fetchReplay()
        this.setState('open')
        resolve()
      }

      // Every rejection below names its failure class. The boot overlay renders this message verbatim, and
      // a bare connectErrorMessage collapses "server refused the token", "TLS/DNS/refused before open" and
      // "nothing answered" into one sentence nobody can act on (#41566).
      const onError = () => {
        if (settled || this.socket !== socket) {
          return
        }

        settled = true
        cleanup()
        this.setState('error')
        // A browser/renderer 'error' event carries no detail; the class is the message.
        reject(this.connectFailure('WebSocket error before open'))
      }

      // A server that closes during the handshake (auth gate, 4401/4403)
      // may never fire `error`; without this the caller waits out the
      // connect timeout for a verdict the socket already delivered. The
      // permanent close listener above normally already dropped the socket
      // and moved the generation to 'closed'; the branch below only runs
      // when `onSocketClose` intercepted that transition and left the
      // half-open socket bound.
      const onClose = (event: CloseEvent) => {
        if (settled) {
          return
        }

        settled = true
        cleanup()

        if (this.socket === socket) {
          this.socket = null
          this.setState('error')
        }

        reject(
          this.connectFailure(
            `WebSocket closed during handshake: code ${event.code}${event.reason ? ` ${event.reason}` : ''}`
          )
        )
      }

      socket.addEventListener('open', onOpen, { once: true })
      socket.addEventListener('error', onError, { once: true })
      socket.addEventListener('close', onClose, { once: true })

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

          reject(this.connectFailure(`no WebSocket open within ${this.options.connectTimeoutMs} ms`))
        }, this.options.connectTimeoutMs)
      }
    })
  }

  private connectFailure(detail: string): Error {
    return new Error(`${this.options.connectErrorMessage} (${detail})`)
  }

  close(): void {
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
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
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
    if (this.options.replay && this.lastSeenSeq.has(sessionId) && this.socket?.readyState !== WebSocket.OPEN) {
      return Promise.resolve(false)
    }

    return undefined
  }

  /**
   * After a reconnect, ask the gateway to replay every event newer than our
   * per-session watermarks. Replayed frames go through the SAME dispatchEvent
   * path as live frames, gated on seq to avoid dispatching duplicates.
   * Best-effort: failures are swallowed (the next reconnect retries).
   */
  private fetchReplay(): void {
    if (!this.options.replay || this.replayHold || this.lastSeenSeq.size === 0) {
      return
    }

    const replayGeneration = ++this.replayGeneration
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

    // A hung background session must not hold a ready session's transcript.
    for (const [sid, lastSeen] of entries) {
      void this.fetchSessionReplay(sid, lastSeen, replayGeneration)
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

  /** Forget the current socket generation, fail its calls, and go 'closed'. */
  private dropSocket(error: Error): void {
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
