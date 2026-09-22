import type { GatewayEvent } from './gateway-events.js'

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
  params?: GatewayEvent | ServerRequestParams
  result?: unknown
}

/**
 * Params of a server→client request (`tui_gateway/server_requests.py`): the
 * backend asking the renderer a question. `session_id` names the session
 * blocked on the answer; the rest is method-specific.
 */
export interface ServerRequestParams extends Record<string, unknown> {
  session_id?: string
}

/** One inbound server→client request, as handed to a `ServerRequestHandler`. */
export interface ServerRequest<M extends string = string, P extends ServerRequestParams = ServerRequestParams> {
  id: string
  method: M
  params: P
  /**
   * Route the answer back to the backend that asked. Idempotent: the first
   * `respond` (or `fail`) wins; a request re-delivered after a reconnect
   * (`open_requests`) reuses the id, so a stale card answering twice is a
   * no-op on the wire.
   */
  respond: (result: Record<string, unknown>) => void
  /** Answer with a JSON-RPC error (the backend treats it as unanswered). */
  fail: (code: number, message: string) => void
  /**
   * Renderer-side tag set by the owner when a request arrives through a
   * replay (`open_requests`) rather than live; handlers that already show the
   * card can skip re-notifying.
   */
  replayed?: boolean
}

/** Handles one inbound server→client request; return `false` to decline (next handler tries). */
export type ServerRequestHandler = (request: ServerRequest) => boolean | void

const isServerRequestFrame = (frame: JsonRpcFrame): frame is JsonRpcFrame & { id: string; method: string } =>
  typeof frame.id === 'string' && typeof frame.method === 'string' && frame.method !== 'event'

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

/** JSON-RPC "internal error" — used when a server→client request handler throws. */
export const JSON_RPC_INTERNAL_ERROR = -32603

/** A window-scoped server request reached a client that does not host its session. */
export const JSON_RPC_REQUEST_NOT_OWNER = -32004

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
  onUnhandledRequest?: (request: { id: string; method: string; params: ServerRequestParams }) => void
  /**
   * A server→client request handler threw: the owner logs it. The channel has
   * already answered `-32603` so the backend does not wait out its deadline
   * against a crashed client (clarify blocks 3600s).
   */
  onRequestHandlerError?: (error: Error, request: { id: string; method: string; params: ServerRequestParams }) => void
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
  resolve: (value: unknown) => void
  timer?: ReturnType<typeof setTimeout>
}

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
  private readonly pending = new Map<GatewayRequestId, PendingCall>()
  private transport: JsonRpcTransport | null = null
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private heartbeatSequence = 0
  private readonly outstandingPings = new Set<string>()
  private lastLivenessAt = 0
  private readonly requestHandlers: ServerRequestHandler[] = []
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

  request<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = this.options.requestTimeoutMs,
    signal?: AbortSignal,
    notConnectedError: () => Error = () => new Error('gateway not connected')
  ): Promise<T> {
    const transport = this.transport

    if (!transport) {
      return Promise.reject(notConnectedError())
    }

    if (signal?.aborted) {
      return Promise.reject(new DOMException('Aborted', 'AbortError'))
    }

    const id = this.options.createRequestId(++this.nextId)

    return new Promise<T>((resolve, reject) => {
      let onAbort: (() => void) | undefined

      const detachAbort = () => {
        if (onAbort && signal) {
          signal.removeEventListener('abort', onAbort)
        }
      }

      const pending: PendingCall = {
        resolve: value => {
          detachAbort()
          resolve(value as T)
        },
        reject: error => {
          detachAbort()
          reject(error)
        }
      }

      if (timeoutMs > 0) {
        pending.timer = setTimeout(() => {
          if (this.pending.delete(id)) {
            detachAbort()
            // Include the configured timeout so a caller (or a user looking
            // at an error toast) can tell whether the default window fired
            // or a per-call override — e.g. /compress opts into 120s.
            const seconds = Math.round(timeoutMs / 1000)
            reject(new Error(`request timed out after ${seconds}s: ${method}`))
          }
        }, timeoutMs)

        if (this.options.unrefTimers) {
          unrefTimer(pending.timer)
        }
      }

      // Abort drops the pending call immediately (no dangling resolver/timer);
      // server-side cancellation is a separate cooperative RPC where it matters.
      if (signal) {
        onAbort = () => {
          this.clearPending(id)
          detachAbort()
          reject(new DOMException('Aborted', 'AbortError'))
        }

        signal.addEventListener('abort', onAbort, { once: true })
      }

      this.pending.set(id, pending)

      try {
        transport.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
      } catch (error) {
        this.clearPending(id)
        detachAbort()
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  /**
   * Register a handler for server→client requests (clarify, approval, sudo,
   * …). Handlers are tried in registration order until one accepts (returns
   * anything but `false`); an unhandled request is answered `-32601` so the
   * backend never waits out its deadline against a client that cannot answer.
   */
  onRequest(handler: ServerRequestHandler): () => void {
    this.requestHandlers.push(handler)

    return () => {
      const index = this.requestHandlers.indexOf(handler)

      if (index >= 0) {
        this.requestHandlers.splice(index, 1)
      }
    }
  }

  /**
   * Deliver a server request to the handlers. Live frames arrive through
   * `handleFrame`; owners call this directly for `open_requests` returned by a
   * reconnect replay (`replayed: true`) so an unanswered question survives a
   * dropped socket.
   */
  deliverRequest(id: string, method: string, params: ServerRequestParams, replayed = false): boolean {
    let settled = false

    const send = (frame: Record<string, unknown>) => {
      if (settled) {
        return
      }

      settled = true

      try {
        this.transport?.send(JSON.stringify({ jsonrpc: '2.0', id, ...frame }))
      } catch {
        // The generation is gone; the backend withdraws the request itself (timeout / reconnect replay).
      }
    }

    const request: ServerRequest = {
      id,
      method,
      params,
      replayed,
      respond: result => send({ result }),
      fail: (code, message) => send({ error: { code, message } })
    }

    for (const handler of this.requestHandlers) {
      let accepted: boolean | void

      try {
        accepted = handler(request)
      } catch (error) {
        // A crashing handler must not leave the backend waiting out its full
        // deadline (clarify blocks 3600s): answer -32603 and stop. The `send`
        // guard makes this a no-op if the handler already responded.
        request.fail(JSON_RPC_INTERNAL_ERROR, `server request handler crashed: ${method}`)
        this.options.onRequestHandlerError?.(error instanceof Error ? error : new Error(String(error)), {
          id,
          method,
          params
        })

        return false
      }

      if (accepted !== false) {
        return true
      }
    }

    request.fail(JSON_RPC_METHOD_NOT_FOUND, `no handler for server request: ${method}`)
    this.options.onUnhandledRequest?.({ id, method, params })

    return false
  }

  private deliverOpenRequests(result: unknown): void {
    const open = (result as { open_requests?: unknown } | null)?.open_requests

    if (!Array.isArray(open)) {
      return
    }

    for (const entry of open as Array<{ id?: unknown; method?: unknown; params?: unknown }>) {
      if (typeof entry?.id === 'string' && typeof entry.method === 'string') {
        const params = entry.params && typeof entry.params === 'object' ? (entry.params as ServerRequestParams) : {}
        this.deliverRequest(entry.id, entry.method, params, true)
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
    let frame: JsonRpcFrame

    try {
      frame = JSON.parse(text) as JsonRpcFrame
    } catch {
      return null
    }

    if (!frame || typeof frame !== 'object') {
      return null
    }

    if (this.options.heartbeatLiveness === 'any-inbound') {
      this.lastLivenessAt = Date.now()
    }

    if (isServerRequestFrame(frame)) {
      const params = frame.params && typeof frame.params === 'object' ? (frame.params as ServerRequestParams) : {}
      this.deliverRequest(frame.id, frame.method, params)

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
          call.reject(jsonRpcErrorFromFrame(frame.error))
        } else {
          // Reconnect contract: `session.resume` / `session.activate` /
          // `session.events.since` answer with `open_requests` — the server→
          // client requests still waiting on this session. They cannot ride
          // the event replay ring (they are not events), so they are re-
          // delivered here, before the caller sees the result, over the very
          // socket that owns them.
          this.deliverOpenRequests(frame.result)
          call.resolve(frame.result)
        }
      }

      return frame
    }

    if (frame.method === 'event' && frame.params && typeof (frame.params as GatewayEvent).type === 'string') {
      if ((frame.params as GatewayEvent).type === 'gateway.ready') {
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
    const call = this.pending.get(id)

    if (call?.timer) {
      clearTimeout(call.timer)
    }

    this.pending.delete(id)
  }

  private rejectAllPending(error: Error): void {
    for (const [id, call] of this.pending) {
      if (call.timer) {
        clearTimeout(call.timer)
      }

      this.pending.delete(id)
      call.reject(error)
    }
  }
}
