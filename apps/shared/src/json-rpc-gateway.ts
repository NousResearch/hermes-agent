import type { GatewayEvent, GatewayEventName } from './gateway-events.js'
import {
  DEFAULT_HEARTBEAT_DEADLINE_MS,
  DEFAULT_HEARTBEAT_INTERVAL_MS,
  type GatewayRequestId,
  type HelloHandshakeResult,
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
  /** An error frame no request owns (the backend's parse-error reply). */
  onUncorrelatedError?: JsonRpcRequestChannelOptions['onUncorrelatedError']
  /**
   * Ordering gate for the `open_requests` an answer re-delivers (#C6): the owner's re-resume of the
   * sessions behind them goes on the wire first, so a card the user answers has a runtime to land
   * on. Never on the critical path — `whenApplicationUsable` does not wait on it.
   */
  beforeOpenRequests?: JsonRpcRequestChannelOptions['beforeOpenRequests']
  /** Return true to intercept the default closed-state transition. */
  onSocketClose?: (event: { code: number }) => boolean | void
  /**
   * C1 reconnect telemetry (t_c1f387b2): fired once per socket generation when the post-open
   * phases are complete (first response landed and, if this generation fetched a reconnect
   * replay, the replay barrier released), and once more — with whatever arrived — if the
   * generation dies before that. Observation only; nothing else reads it.
   */
  onSocketPhases?: (phases: GatewaySocketPhases) => void
  /** Fetch `session.events.since` after a reconnect (default). Off for notification-only feeds whose peer never answers RPCs. */
  replay?: boolean
  requestIdPrefix?: string
  requestTimeoutMs?: number
  socketFactory?: (url: string) => WebSocketLike
  notConnectedErrorMessage?: string
}

/**
 * Durations for one socket generation — the same column names as the C1 phase contract
 * (`deploy/qa_gateway.py phases`, `evals/liveness/wake_latency.py`, the Desktop friction
 * record). Each is measured from socket 'open' unless named otherwise; null = not reached
 * or not applicable to this generation.
 */
export interface GatewaySocketPhases {
  /** connect() call → socket 'open' (DNS/TCP/TLS/upgrade leg). */
  dial_ms: number | null
  /** socket 'open' → the `gateway.ready` event frame. */
  ready_ms: number | null
  /** socket 'open' → the last reconnect replay barrier released (null: no replay was fetched). */
  replay_barrier_ms: number | null
  /** socket 'open' → the first response frame (the first usable application round trip). */
  usable_ms: number | null
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

/**
 * One session's replay window, as either replay transport answers it: the ``session.events.since``
 * result, or one ``client.hello`` ``per_session_replay`` entry (which adds ``session_id`` and the
 * per-session ``error`` its owner reports when that window could not be answered at all).
 */
interface ReplayFrame {
  epoch?: string
  error?: string | null
  events?: GatewayEvent[]
  latest_seq?: number
  replay_epoch?: unknown
  session_id?: string
  snapshot_required?: unknown
  truncated?: boolean
}

/** One socket generation's phase stamps (absolute instants; durations are read off them). */
interface SocketPhaseWatch {
  dialStartedAt: number
  openedAt: number
  readyAt: number | null
  replayRequested: boolean
  replaySettledAt: number | null
  usableAt: number | null
  emitted: boolean
}

/**
 * Most reconnect cursors one client keeps (least recently advanced retire first). Owners keep
 * replay rings for at most 64 sessions (`tui_gateway/event_replay.py::_REPLAY_SESSIONS_MAX`), so a
 * cursor idle behind this many busier sessions is almost never still servable. Retiring it says
 * so the way the owner would: a `session.replay_gap`, after which a pane still showing that
 * session re-snapshots (exactly what `snapshot_required` asks for); nothing is silently lost.
 */
export const REPLAY_CURSOR_MAX = 256

/** The canonical owner's typed refusal for a session it no longer has. */
const replaySessionGone = (error: unknown): boolean =>
  (error as { data?: { reason?: unknown } } | null)?.data?.reason === 'not_found'

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
  /** Canonical authorities number events per session; a session's own epoch outranks the process one. */
  private replayEpochBySession = new Map<string, string>()
  private readonly stateHandlers = new Set<(state: ConnectionState) => void>()
  /** Per-generation C1 phase stamps (t_c1f387b2); null while no generation is open. */
  private phaseWatch: SocketPhaseWatch | null = null
  /** The URL of the socket generation being dialed (what ``helloBackendUrl`` is compared against). */
  private dialedUrl: string | null = null
  /**
   * The URL whose ``gateway.ready`` advertised ``resumed_handshake`` — the backend serves the folded
   * handshake. Remembered so the NEXT reconnect can arm its replay for the fold (one request carrying
   * every watermark) instead of racing it with one ``session.events.since`` per session. Null until a
   * ready frame proves it, so every other backend keeps today's reconnect path unchanged.
   */
  private helloBackendUrl: string | null = null
  /**
   * Replay generation whose per-session requests wait for the ready frame because this reconnect was
   * armed for the fold (``client.hello``) — the fold's answer fills the same hold, and a backend that
   * answers it ``-32601`` (or whose ready does not advertise it after all) gets the fan-out instead.
   */
  private deferredReplay: number | null = null
  // #C3: code waiting for the generation's application-usable point (see whenApplicationUsable).
  private applicationUsableWaiters: Array<() => void> = []
  private dialStartedAt = 0
  private readonly phaseHandlers = new Set<(phases: GatewaySocketPhases) => void>()
  private readonly options: Required<
    Omit<
      GatewayClientOptions,
      'onRequestHandlerError' | 'onUncorrelatedError' | 'onUnhandledRequest' | 'onSocketPhases' | 'socketFactory' | 'beforeOpenRequests'
    >
  > &
    Pick<
      GatewayClientOptions,
      'onRequestHandlerError' | 'onUncorrelatedError' | 'onUnhandledRequest' | 'onSocketPhases' | 'socketFactory' | 'beforeOpenRequests'
    >

  /**
   * This client is the Desktop app on the backend's own machine, read at each socket open
   * (`JsonRpcRequestChannelOptions.desktopLocal`). Only the Desktop's client knows; others are not.
   */
  protected isDesktopLocal(): boolean {
    return false
  }

  constructor(options: GatewayClientOptions = {}) {
    this.options = {
      beforeOpenRequests: options.beforeOpenRequests,
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
      onSocketPhases: options.onSocketPhases,
      onUncorrelatedError: options.onUncorrelatedError,
      onUnhandledRequest: options.onUnhandledRequest,
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      socketFactory: options.socketFactory
    }
    this.channel = new JsonRpcRequestChannel({
      beforeOpenRequests: this.options.beforeOpenRequests,
      createRequestId: this.options.createRequestId,
      desktopLocal: () => this.isDesktopLocal(),
      heartbeatDeadlineMs: this.options.heartbeatDeadlineMs,
      heartbeatIntervalMs: this.options.heartbeatIntervalMs,
      // Desktop/web and the TUI alike count any inbound frame as liveness
      // (#115251): streamed deltas are life; only a silent drop trips the
      // deadline.
      heartbeatLiveness: 'any-inbound',
      helloHandshake: {
        params: () => this.helloParams(),
        onResult: result => this.applyHelloFold(result),
        onUnsupported: () => this.fanOutDeferredReplay()
      },
      onEvent: event => this.handleEvent(event),
      onFirstResponse: () => this.markUsable(),
      onHeartbeatFailure: error => this.invalidate(error.message),
      onRequestHandlerError: this.options.onRequestHandlerError,
      onUncorrelatedError: this.options.onUncorrelatedError,
      onUnhandledRequest: this.options.onUnhandledRequest,
      requestTimeoutMs: this.options.requestTimeoutMs
    })
  }

  get connectionState(): ConnectionState {
    return this.state
  }

  /** The connected canonical owner admits `clarify_batch` on `prompt.submit` (`client.capabilities`). */
  protected get backendClarifyBatch(): boolean {
    return this.channel.clarifyBatch
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

    this.dialStartedAt = Date.now()

    if (this.dialedUrl !== wsUrl) {
      // A different backend: what the previous one advertised says nothing about this one.
      this.helloBackendUrl = null
    }

    this.dialedUrl = wsUrl
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
        // Open before attach: the channel's boot capabilities call rides the first
        // response after this stamp, so it must land inside this generation's watch.
        this.phaseWatch = {
          dialStartedAt: this.dialStartedAt,
          emitted: false,
          openedAt: Date.now(),
          readyAt: null,
          replayRequested: false,
          replaySettledAt: null,
          usableAt: null
        }
        this.channel.attach(transport)
        // Install session barriers before open listeners can start history
        // reads. Replay stays fire-and-forget; connect latency is unchanged.
        this.startReplay()
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

  /**
   * C1 reconnect telemetry (t_c1f387b2): receive one GatewaySocketPhases per socket generation
   * (complete, or partial when the generation died before finishing its open handshake chain).
   */
  onSocketPhases(handler: (phases: GatewaySocketPhases) => void): () => void {
    this.phaseHandlers.add(handler)

    return () => this.phaseHandlers.delete(handler)
  }

  /** First response frame of the current generation: the socket is application-usable. */
  private markUsable(): void {
    const watch = this.phaseWatch

    if (watch && watch.usableAt === null) {
      watch.usableAt = Date.now()
      this.maybeEmitSocketPhases()
      this.settleApplicationUsableWaiters()
    }
  }

  /** All reconnect replay barriers released (or revoked by an epoch change / drop). */
  private markReplayBarrierReleased(): void {
    const watch = this.phaseWatch

    if (watch && watch.replayRequested && watch.replaySettledAt === null) {
      watch.replaySettledAt = Date.now()
      this.maybeEmitSocketPhases()
      this.settleApplicationUsableWaiters()
    }
  }

  /**
   * Resolve when the current socket generation is APPLICATION-USABLE: the first
   * server response has arrived and every replay barrier opened at connect has
   * settled (there was none to open). This is the composer's gate — frames flow,
   * and reads parked behind the transcript barrier have been flushed — and it is
   * exactly the condition C1's per-generation phase emission reports, so waiting
   * here and reading the phases cannot disagree (#C3: post-connect list refreshes
   * moved behind this seam instead of serializing the reconnect path).
   *
   * If the generation dies before reaching usable, waiters resolve on the next
   * generation that does — a refresh that waits for a live socket is correct;
   * one that fires onto a dead one would only fail its RPC anyway.
   */
  whenApplicationUsable(): Promise<void> {
    if (this.isApplicationUsable()) {
      return Promise.resolve()
    }

    return new Promise(resolve => {
      this.applicationUsableWaiters.push(resolve)
    })
  }

  private isApplicationUsable(): boolean {
    const watch = this.phaseWatch

    return watch !== null && watch.usableAt !== null && (!watch.replayRequested || watch.replaySettledAt !== null)
  }

  private settleApplicationUsableWaiters(): void {
    if (!this.isApplicationUsable()) {
      return
    }

    for (const resolve of this.applicationUsableWaiters.splice(0)) {
      resolve()
    }
  }

  /** Emit once per generation when usable has landed and no replay barrier is still outstanding. */
  private maybeEmitSocketPhases(): void {
    const watch = this.phaseWatch

    if (!watch || watch.emitted || watch.usableAt === null || (watch.replayRequested && watch.replaySettledAt === null)) {
      return
    }

    watch.emitted = true
    this.emitSocketPhases(watch)
  }

  private emitSocketPhases(watch: SocketPhaseWatch): void {
    const phases: GatewaySocketPhases = {
      dial_ms: Math.max(0, watch.openedAt - watch.dialStartedAt),
      ready_ms: watch.readyAt === null ? null : Math.max(0, watch.readyAt - watch.openedAt),
      replay_barrier_ms:
        watch.replayRequested && watch.replaySettledAt !== null
          ? Math.max(0, watch.replaySettledAt - watch.openedAt)
          : null,
      usable_ms: watch.usableAt === null ? null : Math.max(0, watch.usableAt - watch.openedAt)
    }

    this.options.onSocketPhases?.(phases)

    for (const handler of this.phaseHandlers) {
      handler(phases)
    }
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
      if (this.phaseWatch && this.phaseWatch.readyAt === null) {
        this.phaseWatch.readyAt = Date.now()
      }

      // This backend serves the folded handshake: remember it so the NEXT reconnect arms its replay
      // for the fold. The channel already opened this generation with `client.hello` (the frame this
      // one answers is the ready frame it read) or, when the flag is absent, with the capabilities
      // probe — in which case the windows the fold would have carried go out one session at a time.
      if (event.payload?.resumed_handshake === true) {
        this.helloBackendUrl = this.dialedUrl
      } else {
        this.fanOutDeferredReplay()
      }

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

    // Parked frames adopt their epoch when flushed, after the replayed gap.
    if (sid) { this.adoptSessionReplayEpoch(sid, event.replay_epoch) }
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
      this.advanceCursor(sid, seq)
    }
  }

  /** Move a session's cursor to `seq` as its most recent; retire the least recently advanced. */
  private advanceCursor(sid: string, seq: number): void {
    this.lastSeenSeq.delete(sid)
    this.lastSeenSeq.set(sid, seq)

    for (const oldest of this.lastSeenSeq.keys()) {
      if (this.lastSeenSeq.size <= REPLAY_CURSOR_MAX) { break }

      // A replay in flight still owns its cursor; it is retired by a later advance.
      if (this.replayHold?.has(oldest)) { continue }

      const latest = this.lastSeenSeq.get(oldest)
      const epoch = this.replayEpochBySession.get(oldest)
      this.lastSeenSeq.delete(oldest)
      this.replayEpochBySession.delete(oldest)
      this.dispatchEvent({ type: 'session.replay_gap', session_id: oldest, payload: { replay_epoch: epoch, latest_seq: latest } })
    }
  }

  /** Test/telemetry hook: current last-seen seq map snapshot. */
  getSeqWatermarks(): Record<string, number> {
    return Object.fromEntries(this.lastSeenSeq)
  }

  /**
   * Forget one session's replay cursor and its epoch qualifier (host-side terminal verdict: the
   * gateway does not have this session — `session.resume` answered `not_found`, so nothing will ever
   * replay under that cursor). Without this a reconnect re-asks for a window that cannot exist, for
   * the client's whole lifetime. Mirrors what the `not_found` branch of a replay answer does.
   */
  forgetReplayCursor(sessionId: string): void {
    this.lastSeenSeq.delete(sessionId)
    this.replayEpochBySession.delete(sessionId)
  }

  /**
   * Continue `from`'s connection instead of cold-booting over it (#C2a wake).
   * The replay cursors are per-instance, so a successor that never inherited
   * them never asks for the window between the old socket's death and the new
   * dial — every event in it is simply lost. The cursor maps are SHARED, not
   * copied: one watermark governs both generations, so whichever socket
   * delivers a frame first advances it and the other generation's copy is
   * dropped — here, by the replay gate's watermark check — and again at the
   * app's cross-socket dedupe. Call before `connect()`: the open-time
   * `fetchReplay()` reads the cursors this adopts.
   */
  inheritReplayState(from: JsonRpcGatewayClient): void {
    if (from === this) {
      return
    }

    this.lastSeenSeq = from.lastSeenSeq
    this.replayEpochBySession = from.replayEpochBySession
    this.replayEpoch = from.replayEpoch
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
   * Install this generation's replay barriers and start the fetch loop, unless the backend is known
   * to serve the folded handshake.
   *
   * Every reconnect with watermarks arms the same per-session hold (live frames for those sessions
   * park behind it, and each session's barrier resolves when its window lands). What differs is how
   * the windows are ASKED for: a backend whose earlier ready frame advertised `resumed_handshake`
   * gets the fold — the client's first frame carries the watermarks and the ONE answer carries every
   * window — so the per-session requests wait for that ready frame instead of racing it. Any other
   * backend (an older one, a transport with no session authority, a harness that sends no ready at
   * all) keeps today's fire-and-forget fan-out from socket open.
   */
  private startReplay(): void {
    const generation = this.armReplayHold()

    if (generation === null) {
      return
    }

    if (this.helloBackendUrl !== null && this.helloBackendUrl === this.dialedUrl) {
      this.deferredReplay = generation

      return
    }

    this.fetchHeldSessions(generation)
  }

  /**
   * Arm the per-session hold and this generation's replay phase. Returns the replay generation the
   * hold belongs to, or null when there is nothing to replay (replay off, no watermarks to resume
   * from, or a hold already installed).
   */
  private armReplayHold(): number | null {
    if (!this.options.replay || this.replayHold || this.lastSeenSeq.size === 0) {
      return null
    }

    const generation = ++this.replayGeneration
    // Park live frames for the sessions we're about to replay so a frame racing the answer can't
    // dispatch ahead of (or duplicate) the gap events. Sessions without watermarks are unaffected.
    const hold = new Map<string, SessionReplay>()

    for (const [sid] of this.lastSeenSeq) {
      let resolve!: (valid: boolean) => void

      const promise = new Promise<boolean>(settle => {
        resolve = settle
      })

      hold.set(sid, { events: [], promise, resolve })
    }

    this.replayHold = hold

    if (this.phaseWatch) {
      this.phaseWatch.replayRequested = true
    }

    return generation
  }

  /** Fan out one `session.events.since` per held session: the reconnect replay without the fold. */
  private fetchHeldSessions(generation: number): void {
    // A hung background session must not hold a ready session's transcript.
    for (const sid of [...(this.replayHold?.keys() ?? [])]) {
      void this.fetchSessionReplay(sid, this.lastSeenSeq.get(sid) ?? 0, generation)
    }
  }

  /**
   * A reconnect was armed for the fold but it will not happen — the backend answered `client.hello`
   * `-32601`, or its ready frame did not advertise the fold after all. Ask the SAME socket one session
   * at a time; never redial (the reconnect owner would charge the whole dial for a fallback the open
   * socket can absorb).
   */
  private fanOutDeferredReplay(): void {
    const generation = this.deferredReplay

    this.deferredReplay = null

    if (generation !== null && this.replayGeneration === generation) {
      this.fetchHeldSessions(generation)
    }
  }

  /**
   * The folded handshake's params: this client's capabilities, its per-session watermarks and the
   * per-session epochs it holds — the qualifier today's one `session.events.since` per session sends
   * as `replay_epoch` (a legacy endpoint refuses any other key, so it is sent only when we have one).
   * Read as the request goes out, so it carries the live cursors exactly like the fan-out does.
   */
  private helloParams(): Record<string, unknown> {
    const epochs = Object.fromEntries(this.replayEpochBySession)

    return {
      capabilities: { server_requests: true },
      watermarks: Object.fromEntries(this.lastSeenSeq),
      ...(this.replayEpoch ? { replay_epoch: this.replayEpoch } : {}),
      ...(Object.keys(epochs).length > 0 ? { session_epochs: epochs } : {})
    }
  }

  /**
   * Apply the folded answer: every session's window through the SAME path the per-session replay
   * uses (epoch rules, truncated skip, seq-gated dispatch), then release whatever is still parked.
   * A window the owner did not answer for a session still releases that session's barrier, so a
   * reader never waits on a window that will not come.
   */
  private applyHelloFold(result: HelloHandshakeResult): void {
    const generation = this.deferredReplay

    this.deferredReplay = null

    if (generation === null || this.replayGeneration !== generation) {
      return
    }

    const windows = Array.isArray(result.per_session_replay) ? result.per_session_replay : []

    for (const entry of windows) {
      if (this.replayGeneration !== generation) {
        return
      }

      const window = (entry ?? {}) as ReplayFrame
      const sid = typeof window.session_id === 'string' ? window.session_id : ''

      if (sid) {
        this.applyReplayWindow(sid, window, generation)
      }
    }

    for (const sid of [...(this.replayHold?.keys() ?? [])]) {
      if (this.replayGeneration !== generation) {
        return
      }

      this.flushReplayHold(sid, generation)
    }
  }

  /**
   * Apply one session's replay window — the epoch rules, the truncated skip and the seq-gated
   * dispatch every replay path shares. `error` is the owner's per-session refusal when the window
   * could not be answered at all; nothing else in the window is trustworthy then.
   */
  private applyReplayWindow(sid: string, window: ReplayFrame, generation: number): void {
    if (window.error) {
      // `not_found` is an answer, not a lost request: the session is gone (deleted, pruned), nothing
      // will ever replay under that cursor, and keeping it re-asks on every reconnect for the
      // client's whole lifetime. Any other refusal keeps the cursor — the next reconnect re-asks.
      if (window.error === 'not_found') {
        this.lastSeenSeq.delete(sid)
        this.replayEpochBySession.delete(sid)
      }

      return
    }

    if (this.applyReplayEpochs(sid, window)) {
      return
    }

    if (window.truncated === true) {
      this.skipTruncatedWindow(sid, window)

      return
    }

    if (!Array.isArray(window.events)) {
      return
    }

    for (const event of window.events) {
      // Event handlers can synchronously invalidate and replace the socket.
      if (this.replayGeneration !== generation) {
        return
      }

      if (event?.type) {
        this.dispatchIfNewer({ ...event, replayed: true })
      }
    }
  }

  private async fetchSessionReplay(sid: string, lastSeen: number, replayGeneration: number): Promise<void> {
    if (this.replayGeneration !== replayGeneration) {
      return
    }

    try {
      // Only a canonical session's own epoch qualifies the watermark on the wire. The legacy
      // process epoch (gateway.ready) is compared with the answer's `epoch` in applyReplayEpochs.
      // An older legacy `session.events.since` refuses any other key, so sending the process
      // epoch here makes the reconnect replay fail and drops pending approval and clarify requests.
      const sessionEpochBefore = this.replayEpochBySession.get(sid)

      // `open_requests` on the answer are re-delivered by the channel itself.
      const result = await this.request<ReplayFrame>(
        'session.events.since',
        { session_id: sid, last_seen: lastSeen, ...(sessionEpochBefore ? { replay_epoch: sessionEpochBefore } : {}) },
        REPLAY_REQUEST_TIMEOUT_MS
      )

      // The socket that owned this replay was dropped while its requests were
      // settling. Its results and cleanup must not consume the replacement
      // socket's replay window.
      if (this.replayGeneration !== replayGeneration) {
        return
      }

      this.applyReplayWindow(sid, result, replayGeneration)
    } catch (error) {
      // Replay is an optimization over lossy-reconnect; never surface errors. But the owner's
      // `not_found` is an answer, not a lost request: the session is gone (deleted, pruned),
      // nothing will ever replay under that cursor, and keeping it re-asked on every reconnect
      // while each session ever observed stayed in both maps for the client's lifetime.
      if (this.replayGeneration === replayGeneration && replaySessionGone(error)) {
        this.lastSeenSeq.delete(sid)
        this.replayEpochBySession.delete(sid)
      }
    } finally {
      if (this.replayGeneration === replayGeneration) {
        this.flushReplayHold(sid, replayGeneration)
      }
    }
  }

  /** Adopt the replay answer's epoch identity; true when the window must not be dispatched. */
  private applyReplayEpochs(
    sid: string,
    result: { epoch?: string; latest_seq?: number; replay_epoch?: unknown; snapshot_required?: unknown } | undefined
  ): boolean {
    const epoch = result?.epoch
    const sessionEpoch = result?.replay_epoch

    if (result?.snapshot_required === true) {
      // The canonical authority could not replay our window (epoch changed,
      // ring truncated, cursor ahead). Keeping the watermark would make the
      // next reconnect believe nothing was missed; drop it and tell the
      // consumer to re-resume for a snapshot. A legacy backend reports a
      // bare `truncated` instead; that skips the window below (#100122).
      // The session's epoch goes with its cursor: it only ever qualifies that cursor, and the
      // re-resumed session's next live frame re-establishes both. Keeping it leaked one entry per
      // session ever observed on this client.
      this.lastSeenSeq.delete(sid)
      this.replayEpochBySession.delete(sid)
      this.dispatchEvent({ type: 'session.replay_gap', session_id: sid, payload: { replay_epoch: sessionEpoch, latest_seq: result.latest_seq } })

      return true
    }

    if (typeof sessionEpoch === 'string' && sessionEpoch) {
      // Canonical responses also include `epoch`, but it is session-local,
      // not the legacy process identity. Never reset unrelated cursors.
      if (this.adoptSessionReplayEpoch(sid, sessionEpoch)) { return true }
    } else if (typeof epoch === 'string' && epoch && this.replayEpoch && epoch !== this.replayEpoch) {
      // The old cursor no longer describes this process's numbering.
      this.adoptReplayEpoch(epoch)

      return true
    } else if (typeof epoch === 'string' && epoch && !this.replayEpoch) {
      this.replayEpoch = epoch
    }

    return false
  }

  /** Advance the watermark past a truncated replay window without dispatching any of it. */
  private skipTruncatedWindow(sid: string, result: { events?: GatewayEvent[]; latest_seq?: number }): void {
    const seqs = (Array.isArray(result.events) ? result.events : []).map(event => event?.seq)

    const head = Math.max(
      this.lastSeenSeq.get(sid) ?? 0,
      ...[result.latest_seq, ...seqs].filter((seq): seq is number => typeof seq === 'number' && Number.isFinite(seq))
    )

    this.advanceCursor(sid, head)
  }

  /**
   * Dispatch an event only when its seq advances the session watermark.
   * Seq-less events always dispatch (no ordering contract to violate).
   */
  private dispatchIfNewer(event: GatewayEvent): void {
    const sid = event.session_id
    const seq = event.seq

    // Parked frames reach here only after the replay window has been applied.
    if (sid) { this.adoptSessionReplayEpoch(sid, event.replay_epoch) }

    if (sid && typeof seq === 'number' && Number.isFinite(seq)) {
      const prev = this.lastSeenSeq.get(sid) ?? 0

      if (seq <= prev) {
        return
      }

      this.advanceCursor(sid, seq)
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
      // Sessions numbered by their own authority epoch keep their cursors.
      for (const sid of this.lastSeenSeq.keys()) {
        if (!this.replayEpochBySession.has(sid)) { this.lastSeenSeq.delete(sid) }
      }

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
      this.markReplayBarrierReleased()
    }

    replay.resolve(true)
  }

  /** Returns true when an established session numbering has changed. */
  private adoptSessionReplayEpoch(sid: string, epoch: unknown): boolean {
    if (typeof epoch !== 'string' || !epoch) { return false }
    const previous = this.replayEpochBySession.get(sid)
    const changed = previous !== undefined && previous !== epoch

    if (changed) { this.lastSeenSeq.delete(sid) }
    this.replayEpochBySession.set(sid, epoch)

    return changed
  }

  private cancelReplay(readsMayProceed: boolean): Map<string, SessionReplay> | null {
    const hold = this.replayHold
    this.replayGeneration += 1
    this.replayHold = null
    // A fold still waiting for its answer belongs to the revoked generation.
    this.deferredReplay = null

    if (hold) {
      // Revoked barriers are released barriers: waiting reads proceed (or were abandoned).
      this.markReplayBarrierReleased()
    }

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
    // C1: a generation that died before finishing reports what it reached (once).
    const watch = this.phaseWatch

    this.phaseWatch = null

    if (watch && !watch.emitted) {
      watch.emitted = true
      this.emitSocketPhases(watch)
    }

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
