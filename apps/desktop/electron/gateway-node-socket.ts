/**
 * Main-process (Node/undici) WebSocket for remote Hermes gateways.
 *
 * "Test remote" already dials `/api/ws` with `globalThis.WebSocket` in the
 * main process and succeeds. The renderer uses Chromium's WebSocket from a
 * `file://` page — different Origin, HTTP/2 Extended CONNECT, and system
 * proxy — so Test can pass while boot still shows "Could not connect to
 * Hermes gateway". Chat must use the same Node socket the probe uses.
 */

export function parseGatewayWsUrl(raw: unknown): URL | null {
  if (typeof raw !== 'string' || !raw.trim()) {
    return null
  }

  try {
    const parsed = new URL(raw)
    const path = parsed.pathname.replace(/\/+$/, '') || '/'

    if (parsed.protocol !== 'ws:' && parsed.protocol !== 'wss:') {
      return null
    }

    if (!path.endsWith('/api/ws')) {
      return null
    }

    return parsed
  } catch {
    return null
  }
}

export function isLoopbackGatewayHost(hostname: string): boolean {
  const host = hostname.trim().toLowerCase()

  return host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1'
}

/** Remote (non-loopback) `/api/ws` URLs should leave Chromium and dial from main. */
export function shouldDialGatewayFromMain(raw: unknown): boolean {
  const parsed = parseGatewayWsUrl(raw)

  return Boolean(parsed && !isLoopbackGatewayHost(parsed.hostname))
}

export function isAllowedGatewayWsUrl(raw: unknown, allowedHosts: string[]): boolean {
  const parsed = parseGatewayWsUrl(raw)

  if (!parsed) {
    return false
  }

  if (isLoopbackGatewayHost(parsed.hostname)) {
    return true
  }

  const host = parsed.hostname.trim().toLowerCase()

  return allowedHosts.some(allowed => allowed.trim().toLowerCase() === host)
}

export function httpsOriginFromGatewayWsUrl(raw: unknown): string | null {
  const parsed = parseGatewayWsUrl(raw)

  if (!parsed) {
    return null
  }

  return `${parsed.protocol === 'wss:' ? 'https:' : 'http:'}//${parsed.host}`
}

type NodeWebSocketLike = {
  addEventListener: (type: string, listener: (event: any) => void) => void
  close: () => void
  send: (data: string) => void
}

type NodeWebSocketCtor = new (url: string, options?: { headers?: Record<string, string> }) => NodeWebSocketLike

export function openNodeGatewaySocket(
  wsUrl: string,
  options: {
    WebSocketImpl: NodeWebSocketCtor
    headers?: Record<string, string>
    onClose: (code: number, reason: string) => void
    onError: () => void
    onMessage: (data: string) => void
    onOpen: () => void
  }
): { close: () => void; send: (data: string) => void } {
  const headers = options.headers && Object.keys(options.headers).length > 0 ? options.headers : undefined
  const socket = headers
    ? new options.WebSocketImpl(wsUrl, { headers })
    : new options.WebSocketImpl(wsUrl)

  socket.addEventListener('open', () => options.onOpen())
  socket.addEventListener('message', event => {
    options.onMessage(typeof event?.data === 'string' ? event.data : String(event?.data ?? ''))
  })
  socket.addEventListener('close', event => {
    options.onClose(Number(event?.code) || 1005, String(event?.reason || ''))
  })
  socket.addEventListener('error', () => options.onError())

  return {
    close: () => {
      try {
        socket.close()
      } catch {
        // already closed
      }
    },
    send: data => {
      socket.send(data)
    }
  }
}
