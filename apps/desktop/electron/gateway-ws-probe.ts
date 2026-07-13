/**
 * Live WebSocket validation for the remote-gateway "Test remote" button.
 *
 * Background: the desktop boot does two independent things to a remote gateway:
 *
 *   1. The MAIN process hits ``GET /api/status`` over HTTP (token in a header)
 *      to confirm the backend is up. This is what "Test remote" historically
 *      checked, and what the boot logs print as "Remote Hermes backend is
 *      ready".
 *   2. The RENDERER then opens a live WebSocket to ``/api/ws`` (credential in a
 *      query param) via ``gateway.connect()``. The chat surface only works once
 *      THIS succeeds.
 *
 * Those two paths use different processes, transports, and credentials, and the
 * server applies extra guards to the WS upgrade that the HTTP status route never
 * sees (Host/Origin checks, ws-ticket/token auth, peer-IP checks). So a gateway
 * can pass the HTTP status check yet reject the WebSocket — which surfaces to
 * the user as a green "Test remote" followed by an opaque "Could not connect to
 * Hermes gateway" on the boot overlay.
 *
 * This module performs the second half of the check: it actually opens the WS
 * URL and confirms the upgrade is accepted (and isn't immediately torn down by
 * a post-upgrade auth rejection). The ``WebSocketImpl`` is injectable so the
 * unit tests can drive the handshake without a real socket; in production the
 * caller passes the Node/Electron global ``WebSocket``.
 */

const DEFAULT_CONNECT_TIMEOUT_MS = 10_000
// After the upgrade is accepted, a gateway that rejects the credential
// post-handshake closes the socket almost immediately. Wait a short grace
// window: a frame (gateway.ready) or a still-open socket means success; an
// early close means the upgrade was accepted but the session was refused.
const DEFAULT_READY_GRACE_MS = 750

/**
 * Attempt a live WebSocket connection and classify the outcome.
 *
 * @param {string} wsUrl - Fully-formed ws(s):// URL including the credential.
 * @returns {Promise<{ ok: boolean, reason?: string }>}
 */
function probeGatewayWebSocket<T>(
  wsUrl: string,
  options: {
    WebSocketImpl?: any
    connectTimeoutMs?: number
    readyGraceMs?: number
  } = {}
) {
  const WebSocketImpl = options.WebSocketImpl
  const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
  const readyGraceMs = options.readyGraceMs ?? DEFAULT_READY_GRACE_MS

  if (typeof WebSocketImpl !== 'function') {
    return Promise.resolve({
      ok: false,
      reason: 'WebSocket is not available in this runtime.'
    })
  }

  return new Promise<any>(resolve => {
    let settled = false
    let opened = false
    let connectTimer = null
    let graceTimer = null
    let socket

    const clearTimers = () => {
      if (connectTimer !== null) {
        clearTimeout(connectTimer)
        connectTimer = null
      }

      if (graceTimer !== null) {
        clearTimeout(graceTimer)
        graceTimer = null
      }
    }

    const finish = result => {
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

    const onOpen = () => {
      if (settled) {
        return
      }
      opened = true
      // Upgrade accepted. Give the server a brief window to reject the
      // credential post-handshake (early close) before declaring success.
      graceTimer = setTimeout(() => {
        finish({ ok: true })
      }, readyGraceMs)
    }

    const onMessage = () => {
      // Any frame means the gateway accepted us and is talking — unambiguous
      // success, no need to wait out the grace window.
      finish({ ok: true })
    }

    const onError = event => {
      finish({
        ok: false,
        reason: extractErrorReason(event) || 'WebSocket connection failed.'
      })
    }

    const onClose = event => {
      if (settled) {
        return
      }

      if (opened) {
        // Opened, then closed inside the grace window: the upgrade was accepted
        // but the session was refused (e.g. ws-ticket/token rejected, or a
        // server-side Host/Origin guard tripped after accept).
        finish({
          ok: false,
          reason: closeReason(event, 'The gateway accepted the connection then closed it (credential rejected?).')
        })

        return
      }

      finish({
        ok: false,
        reason: closeReason(event, 'The gateway closed the WebSocket before it opened.')
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

function addListener(socket, type, handler) {
  if (typeof socket.addEventListener === 'function') {
    socket.addEventListener(type, handler)

    return
  }

  // Node's global WebSocket implements addEventListener; this fallback keeps the
  // helper usable with the `ws` package's EventEmitter shape too.
  if (typeof socket.on === 'function') {
    socket.on(type, handler)
  }
}

function extractErrorReason(event) {
  if (!event) {
    return ''
  }

  if (event instanceof Error) {
    return event.message
  }
  const err = event.error || event.message

  if (err instanceof Error) {
    return err.message
  }

  if (typeof err === 'string') {
    return err
  }

  return ''
}

function closeReason(event, fallback) {
  const code = event && typeof event.code === 'number' ? event.code : null
  const reason = event && typeof event.reason === 'string' ? event.reason.trim() : ''

  if (code && reason) {
    return `${fallback} (code ${code}: ${reason})`
  }

  if (code) {
    return `${fallback} (code ${code})`
  }

  if (reason) {
    return `${fallback} (${reason})`
  }

  return fallback
}

export { DEFAULT_CONNECT_TIMEOUT_MS, DEFAULT_READY_GRACE_MS, probeGatewayWebSocket }
