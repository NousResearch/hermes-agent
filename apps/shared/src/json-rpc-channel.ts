import {
  type ClarifyParams,
  type JsonValue,
  type RpcMethods,
  SERVER_REQUEST_METHODS,
  type ServerRequestMap
} from './gateway-contract.generated.js'
import type { GatewayEvent } from './gateway-events.js'

export type { JsonValue } from './gateway-contract.generated.js'

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
  params?: JsonValue
  result?: JsonValue
}

/** One inbound server→client request, decoded from the generated method map. */
export interface ServerRequest<M extends keyof ServerRequestMap> {
  readonly id: string
  readonly method: M
  readonly params: ServerRequestMap[M]['params']
  readonly sessionId: string | null
  /** The first `respond` (or `fail`) wins, including after a reconnect replay. */
  respond(result: ServerRequestMap[M]['result']): void
  /** Answer with a JSON-RPC error (the backend treats it as unanswered). */
  fail(error: JsonRpcErrorPayload): void
  /** Replay-tagged requests keep their original id and must not re-notify. */
  readonly replayed?: boolean
}

/** The per-method server-request union produced by the wire decoder. */
export type AnyServerRequest = { [M in keyof ServerRequestMap]: ServerRequest<M> }[keyof ServerRequestMap]

type ServerRequestHandler<M extends keyof ServerRequestMap> = (request: ServerRequest<M>) => void

type ServerRequestHandlers = { [M in keyof ServerRequestMap]?: Set<ServerRequestHandler<M>> }

const handlersFor = <M extends keyof ServerRequestMap>(
  handlers: ServerRequestHandlers,
  method: M
): Set<ServerRequestHandler<M>> => {
  const existing = handlers[method]

  if (existing) {
    // SAFETY: this property is only initialized by handlersFor with the same generated method key.
    return existing as Set<ServerRequestHandler<M>>
  }

  const created = new Set<ServerRequestHandler<M>>()
  // SAFETY: this property is only written by handlersFor with the same generated method key.
  handlers[method] = created as ServerRequestHandlers[M]

  return created
}

const anyServerRequest = <M extends keyof ServerRequestMap>(request: ServerRequest<M>): AnyServerRequest => {
  // SAFETY: the generated ServerRequestMap key narrowed by SERVER_REQUEST_METHODS selects the matching union member.
  return request as AnyServerRequest
}

interface DecodedServerRequest<M extends keyof ServerRequestMap> {
  id: string
  method: M
  params: ServerRequestMap[M]['params']
  rawParams: Record<string, JsonValue>
  replayed: boolean
  sessionId: string | null
}

const isServerRequestFrame = (frame: JsonRpcFrame): frame is JsonRpcFrame & { id: string; method: string } =>
  typeof frame.id === 'string' && typeof frame.method === 'string' && frame.method !== 'event'

const isJsonObject = (value: JsonValue): value is Record<string, JsonValue> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const isServerRequestMethod = (method: string): method is keyof ServerRequestMap =>
  SERVER_REQUEST_METHODS.some(serverRequestMethod => serverRequestMethod === method)

/** Contract-7 adapter: a clarify without `kind` is a batch when it carries `questions`, else a single.
 *  Generic over the caller's view (raw wire object or the generated `ClarifyParams`); the one cast is
 *  the wire boundary itself — the frame is JSON and the generated type claims a discriminator the
 *  older backend does not send. */
export const withClarifyKind = <T extends ClarifyParams | Record<string, JsonValue>>(params: T): T => {
  const wire = params as Record<string, JsonValue | undefined>

  if (wire.kind === 'single' || wire.kind === 'batch') {
    return params
  }

  if (Array.isArray(wire.questions)) {
    return { ...wire, kind: 'batch', answers: wire.answers ?? null } as unknown as T
  }

  return {
    ...wire,
    kind: 'single',
    choices: wire.choices ?? null,
    multi_select: wire.multi_select ?? false
  } as unknown as T
}

const decodeServerRequest = <M extends keyof ServerRequestMap>(
  id: string,
  method: M,
  rawParams: JsonValue,
  replayed: boolean
): DecodedServerRequest<M> => {
  const params = isJsonObject(rawParams) ? rawParams : {}

  return {
    id,
    method,
    params: decodeWire<ServerRequestMap[M]['params']>(params),
    rawParams: params,
    replayed,
    sessionId: typeof params.session_id === 'string' ? params.session_id : null
  }
}

const isGatewayEvent = (value: JsonValue): value is { type: string; [key: string]: JsonValue } =>
  isJsonObject(value) && typeof value.type === 'string'

// SAFETY: the backend validated every frame against the same generated contract before sending it;
// this is the one place wire JSON becomes a generated type.
const decodeWire = <T>(value: JsonValue): T => value as T

const jsonFrame = (text: string): JsonRpcFrame | null => {
  try {
    // SAFETY: JSON parse output is an object, and every consumed field is checked below.
    const frame = JSON.parse(text) as JsonRpcFrame

    return frame && typeof frame === 'object' && !Array.isArray(frame) ? frame : null
  } catch {
    return null
  }
}

const responseResult = (frame: JsonRpcFrame): JsonValue => frame.result ?? null

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

/** JSON-RPC "method not found" (tui_gateway/server.py::dispatch `_err(rid, -32601, …)`). */
export const JSON_RPC_METHOD_NOT_FOUND = -32601
export const JSON_RPC_INVALID_PARAMS = -32602

/** JSON-RPC "internal error" — used when a server→client request handler throws. */
export const JSON_RPC_INTERNAL_ERROR = -32603

/** Map a raw `error` member of a response frame to the typed error every surface inspects. */
export function jsonRpcErrorFromFrame(raw: unknown, fallbackMessage = 'Hermes RPC failed'): JsonRpcGatewayError {
  const err = (raw && typeof raw === 'object' ? raw : {}) as JsonRpcErrorPayload

  return new JsonRpcGatewayError(typeof err.message === 'string' && err.message ? err.message : fallbackMessage, {
    code: typeof err.code === 'number' ? err.code : undefined,
    data: err.data
  })
}

/**
 * Anything that can carry one serialized JSON-RPC frame to the gateway. The
 * channel never learns whether that is a WebSocket, a child's stdin, or a
 * test spy; the owner feeds inbound text back through `handleFrame`.
 */
export interface JsonRpcTransport {
  send(text: string): void
}

export interface JsonRpcRequestChannelOptions {
  createRequestId?: (nextId: number) => GatewayRequestId
  heartbeatDeadlineMs?: number
  heartbeatIntervalMs?: number
  /** Called when the heartbeat deadline passes or a heartbeat send throws; the owner drops the transport. */
  onHeartbeatFailure?: (error: Error) => void
  /** Decoded `event` notification. */
  onEvent?: (event: GatewayEvent) => void
  /**
   * Inbound server→client request nobody handled: the owner logs it. The
   * channel has already answered `-32601` so the backend does not wait out
   * its deadline against a client with no handler.
   */
  onUnhandledRequest?: (request: { id: string; method: string; params: Record<string, JsonValue> }) => void
  /**
   * A server→client request handler threw: the owner logs it. The channel has
   * already answered `-32603` so the backend does not wait out its deadline
   * against a crashed client (clarify blocks 3600s).
   */
  onRequestHandlerError?: (error: Error, request: { id: string; method: string; params: Record<string, JsonValue> }) => void
  requestIdPrefix?: string
  requestTimeoutMs?: number
  /**
   * What resets the heartbeat deadline. `'response'` (default): only a
   * `gateway.ping` pong or a response to one of our requests — a backend
   * whose request loop is wedged but still streams deltas is dead for the
   * caller and must be dropped (the Ink TUI's original contract).
   * `'any-inbound'`: every frame, notifications included (the desktop/web
   * WebSocket client's original contract).
   */
  heartbeatLiveness?: HeartbeatLiveness
  /** `setTimeout`/`setInterval` handles are `unref`'d when the runtime supports it (Node) so a pending call cannot pin the process. */
  unrefTimers?: boolean
}

export type HeartbeatLiveness = 'any-inbound' | 'response'

interface PendingCall {
  reject: (error: Error) => void
  resolve: (value: JsonValue) => void
  timer?: ReturnType<typeof setTimeout>
}

interface PendingRequest {
  call: PendingCall
}

interface RequestOptions {
  notConnectedError: () => Error
  signal?: AbortSignal
  timeoutMs: number
}

interface WireRequest {
  method: string
  params: object
}

interface TypedWireRequest<M extends keyof RpcMethods> extends WireRequest {
  method: M
  params: RpcMethods[M]['params']
}

interface UntypedWireRequest extends WireRequest {
  params: Record<string, JsonValue>
}

const typedWireRequest = <M extends keyof RpcMethods>(
  method: M,
  params: RpcMethods[M]['params']
): TypedWireRequest<M> => ({
  method,
  params
})

const untypedWireRequest = (method: string, params: Record<string, JsonValue>): UntypedWireRequest => ({
  method,
  params
})

const requestOptions = (
  timeoutMs: number,
  signal: AbortSignal | undefined,
  notConnectedError: () => Error
): RequestOptions => ({
  notConnectedError,
  signal,
  timeoutMs
})

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000
// Keepalive + dead-connection detection. A silent drop (macOS sleep, proxy
// idle timeout, VPN reconnect) kills the TCP socket without a `close` event,
// so the client hangs forever (issue #32997). Browser/undici WebSocket does
// not expose an acknowledged ping/pong API, so this uses a small JSON-RPC
// heartbeat that the TUI gateway explicitly answers.
export const DEFAULT_HEARTBEAT_INTERVAL_MS = 15_000
export const DEFAULT_HEARTBEAT_DEADLINE_MS = 45_000
const MAX_OUTSTANDING_PINGS = 8

// Hoisted decoder: attach mode can drive high-frequency binary frames (tool
// deltas, reasoning streams) and a fresh TextDecoder per message is avoidable
// GC pressure; UTF-8 is stateless and frames arrive whole.
const wireDecoder = new TextDecoder()

/** Decode a socket `message.data` (string / ArrayBuffer / view) to text; `null` for anything else. */
export function wireFrameText(raw: unknown): string | null {
  if (typeof raw === 'string') {
    return raw
  }

  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
    return wireDecoder.decode(raw as ArrayBuffer)
  }

  return null
}

const unrefTimer = (timer: unknown) => {
  ;(timer as { unref?: () => void } | undefined)?.unref?.()
}

/**
 * The transport-agnostic half of a JSON-RPC gateway connection: request ids,
 * the pending map with per-call timeouts and AbortSignal, response → typed
 * error mapping, event-notification decoding, and the `gateway.ping`
 * heartbeat. Owners (`JsonRpcGatewayClient` over WebSocket, the Ink TUI over
 * stdio / an attached socket) supply a `JsonRpcTransport` per connection
 * generation and call `handleFrame` for every inbound text frame.
 */
export class JsonRpcRequestChannel {
  private nextId = 0
  private readonly pending = new Map<GatewayRequestId, PendingRequest>()
  private transport: JsonRpcTransport | null = null
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private heartbeatSequence = 0
  private readonly outstandingPings = new Set<string>()
  private lastLivenessAt = 0
  private readonly serverRequestHandlers: ServerRequestHandlers = {}
  private readonly anyServerRequestHandlers = new Set<(request: AnyServerRequest) => void>()
  private readonly options: Required<
    Omit<JsonRpcRequestChannelOptions, 'onEvent' | 'onHeartbeatFailure' | 'onRequestHandlerError' | 'onUnhandledRequest'>
  > &
    Pick<JsonRpcRequestChannelOptions, 'onEvent' | 'onHeartbeatFailure' | 'onRequestHandlerError' | 'onUnhandledRequest'>

  constructor(options: JsonRpcRequestChannelOptions = {}) {
    this.options = {
      createRequestId: options.createRequestId ?? ((nextId: number) => `${options.requestIdPrefix ?? 'r'}${nextId}`),
      heartbeatDeadlineMs: options.heartbeatDeadlineMs ?? DEFAULT_HEARTBEAT_DEADLINE_MS,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      heartbeatLiveness: options.heartbeatLiveness ?? 'response',
      onEvent: options.onEvent,
      onHeartbeatFailure: options.onHeartbeatFailure,
      onRequestHandlerError: options.onRequestHandlerError,
      onUnhandledRequest: options.onUnhandledRequest,
      requestIdPrefix: options.requestIdPrefix ?? 'r',
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      unrefTimers: options.unrefTimers ?? false
    }
  }

  get defaultRequestTimeoutMs(): number {
    return this.options.requestTimeoutMs
  }

  get connected(): boolean {
    return this.transport !== null
  }

  /** Bind a new connection generation. Any previous generation's heartbeat stops; its pending calls are the owner's to reject. */
  attach(transport: JsonRpcTransport): void {
    this.stopHeartbeat()
    this.transport = transport
    this.lastLivenessAt = Date.now()
  }

  /** Drop the transport and fail every in-flight call with `error`. */
  detach(error: Error): void {
    this.stopHeartbeat()
    this.transport = null
    this.rejectAllPending(error)
  }

  /** True while `transport` is the bound generation (owners gate stale socket callbacks on this). */
  owns(transport: JsonRpcTransport): boolean {
    return this.transport === transport
  }

  request<M extends keyof RpcMethods>(
    method: M,
    params: RpcMethods[M]['params'],
    timeoutMs = this.options.requestTimeoutMs,
    signal?: AbortSignal,
    notConnectedError: () => Error = () => new Error('gateway not connected')
  ): Promise<RpcMethods[M]['result']> {
    return this.requestInternal(
      typedWireRequest(method, params),
      requestOptions(timeoutMs, signal, notConnectedError)
    ).then(value => decodeWire<RpcMethods[M]['result']>(value))
  }

  // SAFETY: plugins are third-party code; the method name is not known at compile time.
  requestUntyped(
    method: string,
    params: Record<string, JsonValue>,
    timeoutMs = this.options.requestTimeoutMs,
    signal?: AbortSignal,
    notConnectedError: () => Error = () => new Error('gateway not connected')
  ): Promise<JsonValue> {
    return this.requestInternal(
      untypedWireRequest(method, params),
      requestOptions(timeoutMs, signal, notConnectedError)
    )
  }

  private requestInternal(request: WireRequest, options: RequestOptions): Promise<JsonValue> {
    const transport = this.transport

    if (!transport) {
      return Promise.reject(options.notConnectedError())
    }

    if (options.signal?.aborted) {
      return Promise.reject(new DOMException('Aborted', 'AbortError'))
    }

    const id = this.options.createRequestId(++this.nextId)

    return new Promise((resolve, reject) => {
      let onAbort: (() => void) | undefined

      const detachAbort = () => {
        if (onAbort && options.signal) {
          options.signal.removeEventListener('abort', onAbort)
        }
      }

      const call: PendingCall = {
        resolve: value => {
          detachAbort()
          resolve(value)
        },
        reject: error => {
          detachAbort()
          reject(error)
        }
      }

      if (options.timeoutMs > 0) {
        call.timer = setTimeout(() => {
          if (this.pending.delete(id)) {
            detachAbort()
            // Include the configured timeout so a caller (or a user looking
            // at an error toast) can tell whether the default window fired
            // or a per-call override — e.g. /compress opts into 120s.
            const seconds = Math.round(options.timeoutMs / 1000)
            reject(new Error(`request timed out after ${seconds}s: ${request.method}`))
          }
        }, options.timeoutMs)

        if (this.options.unrefTimers) {
          unrefTimer(call.timer)
        }
      }

      // Abort drops the pending call immediately (no dangling resolver/timer);
      // server-side cancellation is a separate cooperative RPC where it matters.
      if (options.signal) {
        onAbort = () => {
          this.clearPending(id)
          detachAbort()
          reject(new DOMException('Aborted', 'AbortError'))
        }

        options.signal.addEventListener('abort', onAbort, { once: true })
      }

      this.pending.set(id, { call })

      try {
        transport.send(JSON.stringify({ jsonrpc: '2.0', id, ...request }))
      } catch (error) {
        this.clearPending(id)
        detachAbort()
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  /** Register a handler for one server→client request method. */
  onServerRequest<M extends keyof ServerRequestMap>(
    method: M,
    handler: (request: ServerRequest<M>) => void
  ): () => void {
    const handlers = handlersFor(this.serverRequestHandlers, method)
    handlers.add(handler)

    return () => handlers.delete(handler)
  }

  /** Catch every generated server→client request without widening its payload type. */
  onAnyServerRequest(handler: (request: AnyServerRequest) => void): () => void {
    this.anyServerRequestHandlers.add(handler)

    return () => this.anyServerRequestHandlers.delete(handler)
  }

  /** Deliver a live or replayed server request through the typed method table. */
  deliverRequest(id: string, method: string, rawParams: JsonValue, replayed = false): boolean {
    if (!isServerRequestMethod(method)) {
      this.sendServerRequestResponse(id, {
        error: { code: JSON_RPC_METHOD_NOT_FOUND, message: `no handler for server request: ${method}` }
      })
      this.options.onUnhandledRequest?.({ id, method, params: isJsonObject(rawParams) ? rawParams : {} })

      return false
    }

    // The one discriminated server request. A contract-7 backend (current `main`) sends a clarify with no
    // `kind` — `{questions}` for a batch, `{question, choices, multi_select?}` for a single — and it shares
    // the contract number with this client, so no update toast fires: stamp the discriminator here rather
    // than refuse every question of a mixed-version pair. Anything that is not an object stays refused.
    if (method === 'clarify') {
      if (!isJsonObject(rawParams)) {
        this.sendServerRequestResponse(id, {
          error: { code: JSON_RPC_INVALID_PARAMS, message: 'clarify request params must be an object' }
        })

        return false
      }

      return this.deliverDecodedServerRequest(decodeServerRequest(id, method, withClarifyKind(rawParams), replayed))
    }

    return this.deliverDecodedServerRequest(decodeServerRequest(id, method, rawParams, replayed))
  }

  private deliverDecodedServerRequest<M extends keyof ServerRequestMap>(decoded: DecodedServerRequest<M>): boolean {
    let settled = false

    const send = (frame: { error: JsonRpcErrorPayload } | { result: object }) => {
      if (settled) {
        return
      }

      settled = true
      this.sendServerRequestResponse(decoded.id, frame)
    }

    const request: ServerRequest<M> = {
      id: decoded.id,
      method: decoded.method,
      params: decoded.params,
      sessionId: decoded.sessionId,
      replayed: decoded.replayed,
      respond: result => send({ result }),
      fail: error => send({ error })
    }

    // A crashing handler must not leave the backend waiting out its full
    // deadline (clarify blocks 3600s): answer -32603 and stop. The `send`
    // guard makes this a no-op if the handler already responded.
    const invoke = (handler: () => void): boolean => {
      try {
        handler()

        return true
      } catch (error) {
        request.fail({ code: JSON_RPC_INTERNAL_ERROR, message: `server request handler crashed: ${decoded.method}` })
        this.options.onRequestHandlerError?.(error instanceof Error ? error : new Error(String(error)), {
          id: decoded.id,
          method: decoded.method,
          params: decoded.rawParams
        })

        return false
      }
    }

    const handlers = handlersFor(this.serverRequestHandlers, decoded.method)

    for (const handler of handlers) {
      if (!invoke(() => handler(request))) {
        return false
      }
    }

    for (const handler of this.anyServerRequestHandlers) {
      if (!invoke(() => handler(anyServerRequest(request)))) {
        return false
      }
    }

    if (handlers.size || this.anyServerRequestHandlers.size) {
      return true
    }

    request.fail({ code: JSON_RPC_METHOD_NOT_FOUND, message: `no handler for server request: ${decoded.method}` })
    this.options.onUnhandledRequest?.({ id: decoded.id, method: decoded.method, params: decoded.rawParams })

    return false
  }

  private sendServerRequestResponse(id: string, frame: { error: JsonRpcErrorPayload } | { result: object }): void {
    try {
      this.transport?.send(JSON.stringify({ jsonrpc: '2.0', id, ...frame }))
    } catch {
      // The generation is gone; the backend withdraws the request itself (timeout / reconnect replay).
    }
  }

  private deliverOpenRequests(result: JsonValue): void {
    if (!isJsonObject(result) || !Array.isArray(result.open_requests)) {
      return
    }

    for (const entry of result.open_requests) {
      if (isJsonObject(entry) && typeof entry.id === 'string' && typeof entry.method === 'string') {
        this.deliverRequest(entry.id, entry.method, entry.params ?? null, true)
      }
    }
  }

  /**
   * Route one inbound frame: a response settles its pending call, an
   * `event` notification reaches `onEvent`, a server→client request reaches
   * the `onRequest` handlers. Returns the decoded frame so the owner can act
   * on it too (mirror it, record seq, …) or `null` when the text was not
   * JSON or not a JSON object (`null`, a scalar).
   */
  handleFrame(text: string): JsonRpcFrame | null {
    const frame = jsonFrame(text)

    if (!frame) {
      return null
    }

    if (this.options.heartbeatLiveness === 'any-inbound') {
      this.lastLivenessAt = Date.now()
    }

    if (isServerRequestFrame(frame)) {
      this.deliverRequest(frame.id, frame.method, frame.params ?? null)

      return frame
    }

    if (frame.id !== undefined && frame.id !== null) {
      if (typeof frame.id === 'string' && this.outstandingPings.delete(frame.id)) {
        this.lastLivenessAt = Date.now()

        return frame
      }

      const call = this.pending.get(frame.id)

      if (call) {
        this.lastLivenessAt = Date.now()
        this.clearPending(frame.id)

        if (frame.error) {
          call.call.reject(jsonRpcErrorFromFrame(frame.error))
        } else {
          // Reconnect contract: `session.resume` / `session.activate` /
          // `session.events.since` answer with `open_requests` — the server→
          // client requests still waiting on this session. They cannot ride
          // the event replay ring (they are not events), so they are re-
          // delivered here, before the caller sees the result, over the very
          // socket that owns them.
          const result = responseResult(frame)
          this.deliverOpenRequests(result)

          call.call.resolve(result)
        }
      }

      return frame
    }

    if (frame.method === 'event' && frame.params && isGatewayEvent(frame.params)) {
      // SAFETY: event envelopes carry generated payload shapes; this boundary confirms their discriminator.
      if (frame.params.type === 'gateway.ready') {
        this.advertiseCapabilities()
      }

      this.options.onEvent?.(frame.params as GatewayEvent)
    }

    return frame
  }

  /**
   * Tell the backend, once per connection generation, that this client answers
   * server→client requests (a handler result or `-32601`). Without it a
   * backend treats a WebSocket client as a build that predates server requests
   * and fails every clarify/approval/… for it immediately instead of stalling
   * the agent for the full deadline. An older backend answers `-32601` here;
   * that is ignored.
   */
  private advertiseCapabilities(): void {
    this.request('client.capabilities', { server_requests: true }).catch(() => undefined)
  }

  /**
   * Begin the `gateway.ping` keepalive on the bound transport. Only call when
   * `gateway.ready.heartbeat` advertised support — an older backend would
   * answer with -32601 and never count as alive. What counts as liveness is
   * `heartbeatLiveness`; a full deadline without it drops the transport.
   */
  startHeartbeat(): void {
    this.stopHeartbeat()
    this.lastLivenessAt = Date.now()

    const transport = this.transport

    if (!transport || this.options.heartbeatIntervalMs <= 0 || this.options.heartbeatDeadlineMs <= 0) {
      return
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.transport !== transport) {
        return
      }

      if (Date.now() - this.lastLivenessAt >= this.options.heartbeatDeadlineMs) {
        this.failHeartbeat(new Error('WebSocket heartbeat acknowledgement timed out'))

        return
      }

      const id = `heartbeat-${++this.heartbeatSequence}`
      this.outstandingPings.add(id)

      // In 'any-inbound' mode a backend that streams but never pongs keeps
      // the transport alive indefinitely; forget stale ping ids so the set
      // cannot grow with it.
      if (this.outstandingPings.size > MAX_OUTSTANDING_PINGS) {
        this.outstandingPings.delete(this.outstandingPings.values().next().value as string)
      }

      try {
        transport.send(JSON.stringify({ jsonrpc: '2.0', id, method: 'gateway.ping', params: {} }))
      } catch (error) {
        this.failHeartbeat(error instanceof Error ? error : new Error(String(error)))
      }
    }, this.options.heartbeatIntervalMs)

    if (this.options.unrefTimers) {
      unrefTimer(this.heartbeatTimer)
    }
  }

  stopHeartbeat(): void {
    this.outstandingPings.clear()

    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private failHeartbeat(error: Error): void {
    this.stopHeartbeat()
    this.options.onHeartbeatFailure?.(error)
  }

  private clearPending(id: GatewayRequestId): void {
    const pending = this.pending.get(id)

    if (pending?.call.timer) {
      clearTimeout(pending.call.timer)
    }

    this.pending.delete(id)
  }

  private rejectAllPending(error: Error): void {
    for (const [id, pending] of this.pending) {
      if (pending.call.timer) {
        clearTimeout(pending.call.timer)
      }

      this.pending.delete(id)
      pending.call.reject(error)
    }
  }
}
