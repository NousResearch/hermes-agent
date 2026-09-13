import type { GatewayEvent, GatewayEventName } from './gateway-events.js'
import {
  DEFAULT_HEARTBEAT_DEADLINE_MS,
  DEFAULT_HEARTBEAT_INTERVAL_MS,
  type GatewayRequestId,
  JsonRpcRequestChannel,
  type JsonRpcTransport,
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
  if (typeof value !== 'string') {return false}

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

export class JsonRpcGatewayClient {
  private socket: WebSocketLike | null = null
  private state: ConnectionState = 'idle'
  private readonly channel: JsonRpcRequestChannel
  private readonly events = new GatewayEventHub()
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
      replay: options.replay ?? true,
      requestIdPrefix: options.requestIdPrefix ?? 'r',
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      socketFactory: options.socketFactory
    }
    this.channel = new JsonRpcRequestChannel({
      createRequestId: this.options.createRequestId,
      heartbeatDeadlineMs: this.options.heartbeatDeadlineMs,
      heartbeatIntervalMs: this.options.heartbeatIntervalMs,
      // Desktop/web have always counted any inbound frame as liveness; the
      // TUI (stdio/attach owner) keeps its stricter pong-based contract.
      heartbeatLiveness: 'any-inbound',
      onEvent: event => this.handleEvent(event),
      onHeartbeatFailure: error => this.invalidate(error.message),
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

      // A server that closes during the handshake (auth gate, 4401/4403)
      // may never fire `error`; without this the caller waits out the
      // connect timeout for a verdict the socket already delivered. The
      // permanent close listener above normally already dropped the socket
      // and moved the generation to 'closed'; the branch below only runs
      // when `onSocketClose` intercepted that transition and left the
      // half-open socket bound.
      const onClose = () => {
        if (settled) {
          return
        }

        settled = true
        cleanup()

        if (this.socket === socket) {
          this.socket = null
          this.setState('error')
        }

        reject(new Error(this.options.connectErrorMessage))
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

          reject(new Error(this.options.connectErrorMessage))
        }, this.options.connectTimeoutMs)
      }
    })
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

    return this.channel.request<T>(method, params, timeoutMs, signal, () => new Error(this.options.notConnectedErrorMessage))
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
      this.replayHold.get(sid)?.push(event)

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
   * After a reconnect, ask the gateway to replay every event newer than our
   * per-session watermarks. Replayed frames go through the SAME dispatchEvent
   * path as live frames — dedupe happens naturally because recordSeq ignores
   * non-increasing seqs and downstream stores key on event identity.
   * Best-effort: failures are swallowed (the next reconnect retries).
   */
  private async fetchReplay(): Promise<void> {
    if (!this.options.replay || this.replayInFlight || this.lastSeenSeq.size === 0) {
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

  /** Forget the current socket generation, fail its calls, and go 'closed'. */
  private dropSocket(error: Error): void {
    this.socket = null
    this.channel.detach(error)
    this.setState('closed')
  }

  private dispatchEvent(event: GatewayEvent): void {
    this.events.dispatch(event)
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
