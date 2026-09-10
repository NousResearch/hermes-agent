/**
 * JSON-RPC round-trip validation for the desktop's local backend boot.
 *
 * `probeGatewayWebSocket` proves the /api/ws upgrade is ACCEPTED — but a
 * backend left half-updated by an interrupted `hermes update` (Windows venv
 * lock aborts the swap mid-flight, #92927) can still accept the upgrade while
 * its JSON-RPC dispatcher is dead. Boot then declares `backend.ready` off
 * HTTP+WS readiness alone and the renderer boots into a silent shell: empty
 * BOTS pane, `window.hermesDesktop is undefined` in the served dashboard, no
 * error surfaced.
 *
 * This probe performs the one round-trip the ENTIRE desktop surface depends
 * on — a JSON-RPC request and a reply carrying our id. The reply may be a
 * result OR a JSON-RPC error (e.g. -32601 for an unknown method): both prove
 * the dispatcher processed the request, so this stays valid across older
 * gateways without depending on any specific method's availability.
 *
 * The `WebSocketImpl` is injectable so unit tests can drive the handshake
 * without a real socket; in production the caller passes the Node/Electron
 * global `WebSocket`, exactly like gateway-ws-probe.ts.
 */

const DEFAULT_CONNECT_TIMEOUT_MS = 10_000
// Generous for a local backend: the first dispatch after spawn can contend
// with the startup import storm, and this probe's false-positive (failing a
// healthy boot) is far worse than its false-negative.
const DEFAULT_REPLY_TIMEOUT_MS = 8_000

// A string id is unambiguous and JSON-RPC servers MUST echo ids verbatim.
const PROBE_REQUEST_ID = 'desktop-boot-rpc-probe'

export interface GatewayRpcProbeOptions {
  WebSocketImpl?: any
  connectTimeoutMs?: number
  replyTimeoutMs?: number
  method?: string
  params?: Record<string, unknown>
  requestId?: string
}

interface ProbeResult {
  ok: boolean
  reason?: string
  /**
   * Machine-readable failure class. `websocket-unavailable` marks a
   * CAPABILITY problem (the runtime has no WebSocket constructor), which must
   * never be reported as a torn install — see rpcProbeBootError().
   */
  code?: 'websocket-unavailable'
}

/**
 * Open `wsUrl`, send one JSON-RPC request, and require a reply frame whose
 * `id` matches ours (result or JSON-RPC error both qualify).
 */
function probeGatewayRpc(wsUrl: string, options: GatewayRpcProbeOptions = {}): Promise<ProbeResult> {
  const WebSocketImpl = options.WebSocketImpl
  const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
  const replyTimeoutMs = options.replyTimeoutMs ?? DEFAULT_REPLY_TIMEOUT_MS
  const method = options.method ?? 'session.list'
  const params = options.params ?? {}
  const requestId = options.requestId ?? PROBE_REQUEST_ID

  if (typeof WebSocketImpl !== 'function') {
    return Promise.resolve({
      ok: false,
      code: 'websocket-unavailable',
      reason: 'WebSocket is not available in this runtime.'
    })
  }

  return new Promise<ProbeResult>(resolve => {
    let settled = false
    let opened = false
    let connectTimer: ReturnType<typeof setTimeout> | null = null
    let replyTimer: ReturnType<typeof setTimeout> | null = null
    let socket: any

    const clearTimers = () => {
      if (connectTimer !== null) {
        clearTimeout(connectTimer)
        connectTimer = null
      }

      if (replyTimer !== null) {
        clearTimeout(replyTimer)
        replyTimer = null
      }
    }

    const finish = (result: ProbeResult) => {
      if (settled) {
        return
      }

      settled = true
      clearTimers()

      try {
        socket?.close?.()
      } catch {
        // ignore — best effort teardown
      }

      resolve(result)
    }

    try {
      socket = new WebSocketImpl(wsUrl)
    } catch (error) {
      finish({
        ok: false,
        reason: error instanceof Error ? error.message : String(error)
      })

      return
    }

    const sendRequest = () => {
      try {
        socket.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id: requestId,
            method,
            params
          })
        )
      } catch (error) {
        finish({
          ok: false,
          reason: error instanceof Error ? error.message : String(error)
        })

        return
      }

      // The dispatcher's reply (any frame carrying our id) is the whole
      // contract; nothing else may end the wait.
      replyTimer = setTimeout(() => {
        finish({
          ok: false,
          reason: `Timed out after ${replyTimeoutMs}ms waiting for a JSON-RPC reply to "${method}".`
        })
      }, replyTimeoutMs)
    }

    const onOpen = () => {
      if (settled) {
        return
      }

      opened = true
      // The socket opened: the connect deadline has done its job. Disarm it
      // so a slow reply (or a grace window longer than connectTimeoutMs)
      // can't be misreported as a connect timeout on an open connection.
      if (connectTimer !== null) {
        clearTimeout(connectTimer)
        connectTimer = null
      }
      sendRequest()
    }

    const onMessage = (event: any) => {
      if (settled) {
        return
      }

      let frame: any

      try {
        frame = JSON.parse(typeof event?.data === 'string' ? event.data : String(event?.data ?? ''))
      } catch {
        return
      }

      // Events (`gateway.ready`, `session.info`, …) carry no id and replies to
      // other requests carry someone else's id — neither proves OUR round-trip.
      if (frame?.id === requestId) {
        finish({ ok: true })
      }
    }

    const onError = (event: any) => {
      finish({
        ok: false,
        reason:
          (event instanceof Error ? event.message : event?.error?.message || event?.message) ||
          'WebSocket connection failed.'
      })
    }

    const onClose = (event: any) => {
      if (settled) {
        return
      }

      if (opened) {
        finish({
          ok: false,
          reason: `The gateway closed the WebSocket before replying to "${method}" (code ${event?.code ?? 'unknown'}).`
        })

        return
      }

      finish({
        ok: false,
        reason: 'The gateway closed the WebSocket before it opened.'
      })
    }

    addListener(socket, 'open', onOpen)
    addListener(socket, 'message', onMessage)
    addListener(socket, 'error', onError)
    addListener(socket, 'close', onClose)

    if (connectTimeoutMs > 0) {
      connectTimer = setTimeout(() => {
        finish({
          ok: false,
          reason: `Timed out after ${connectTimeoutMs}ms waiting for the WebSocket to open.`
        })
      }, connectTimeoutMs)
    }
  })
}

function addListener(socket: any, type: string, handler: (event: any) => void) {
  if (typeof socket.addEventListener === 'function') {
    socket.addEventListener(type, handler)

    return
  }

  // The `ws` package's EventEmitter shape, same courtesy as gateway-ws-probe.
  if (typeof socket.on === 'function') {
    socket.on(type, handler)
  }
}

// Stable failure codes attached to the boot error (and carried to the
// renderer as DesktopBootProgress.errorCode, see src/global.d.ts) so the
// recovery overlay can key on the code instead of string-matching the
// message — the same structured-classification pattern as
// isCloudBackendDown (#85335).
export const RPC_PROBE_FAILED_ERROR_CODE = 'gateway.rpc-probe-failed'
export const RPC_PROBE_UNAVAILABLE_ERROR_CODE = 'gateway.rpc-unavailable'

// SINGLE SOURCE for the repair guidance both local boot paths surface when
// the backend is reachable but its JSON-RPC dispatcher never answered
// (#92927: an interrupted `hermes update` can leave the dispatcher dead while
// HTTP/WS readiness still passes). Both call sites in main.ts build their
// boot error through rpcProbeBootError() so this copy cannot drift between
// the pool-spawn boot and the primary boot.
const TORN_INSTALL_REPAIR_GUIDANCE =
  'The install may be broken by an interrupted update — repair with: hermes desktop --force-build, or re-run: hermes update'

/**
 * Build the boot error for a failed gateway RPC probe. One message shape for
 * every failure (reachable backend, dead dispatcher), except the
 * `websocket-unavailable` case: a missing WebSocket constructor is a
 * CAPABILITY problem, not a torn install, so it gets a distinct message and
 * a distinct stable code — no `hermes update` advice for a problem the
 * updater cannot fix.
 */
export function rpcProbeBootError(backendLabel: string, probe: ProbeResult): Error & { code: string } {
  if (probe.code === 'websocket-unavailable') {
    const error = new Error(
      `${backendLabel} is reachable but this runtime cannot open a WebSocket to verify its JSON-RPC gateway (${probe.reason ?? 'WebSocket is not available in this runtime.'}).`
    ) as Error & { code: string }
    error.code = RPC_PROBE_UNAVAILABLE_ERROR_CODE

    return error
  }

  const error = new Error(
    `${backendLabel} is reachable but its JSON-RPC gateway did not answer (${probe.reason ?? 'no reply received'}). ` +
      TORN_INSTALL_REPAIR_GUIDANCE
  ) as Error & { code: string }
  error.code = RPC_PROBE_FAILED_ERROR_CODE

  return error
}

export {
  DEFAULT_CONNECT_TIMEOUT_MS,
  DEFAULT_REPLY_TIMEOUT_MS,
  PROBE_REQUEST_ID,
  probeGatewayRpc,
  type ProbeResult
}
