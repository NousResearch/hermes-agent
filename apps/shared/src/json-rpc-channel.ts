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

/** JSON-RPC "method not found" (tui_gateway/server.py::dispatch `_err(rid, -32601, …)`). */
export const JSON_RPC_METHOD_NOT_FOUND = -32601

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
  private readonly options: Required<Omit<JsonRpcRequestChannelOptions, 'onEvent' | 'onHeartbeatFailure'>> &
    Pick<JsonRpcRequestChannelOptions, 'onEvent' | 'onHeartbeatFailure'>

  constructor(options: JsonRpcRequestChannelOptions = {}) {
    this.options = {
      createRequestId: options.createRequestId ?? ((nextId: number) => `${options.requestIdPrefix ?? 'r'}${nextId}`),
      heartbeatDeadlineMs: options.heartbeatDeadlineMs ?? DEFAULT_HEARTBEAT_DEADLINE_MS,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      heartbeatLiveness: options.heartbeatLiveness ?? 'response',
      onEvent: options.onEvent,
      onHeartbeatFailure: options.onHeartbeatFailure,
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
   * Route one inbound frame: a response settles its pending call, an
   * `event` notification reaches `onEvent`. Returns the decoded frame so the
   * owner can act on it too (mirror it, record seq, …) or `null` when the
   * text was not JSON or not a JSON object (`null`, a scalar).
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
          call.resolve(frame.result)
        }
      }

      return frame
    }

    if (frame.method === 'event' && frame.params && typeof frame.params.type === 'string') {
      this.options.onEvent?.(frame.params)
    }

    return frame
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
