import crypto from 'node:crypto'
import http from 'node:http'
import https from 'node:https'
import type { Socket } from 'node:net'

const DEFAULT_CONNECT_TIMEOUT_MS = 10_000
const DEFAULT_READY_GRACE_MS = 750
const WEBSOCKET_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

interface ProbeOptions {
  connectTimeoutMs?: number
  headers?: Record<string, string>
  readyGraceMs?: number
}

interface ProbeResult {
  ok: boolean
  reason?: string
}

/**
 * Probe a WebSocket upgrade with custom HTTP headers.
 *
 * The WHATWG WebSocket used by Electron's main process cannot attach request
 * headers. A small RFC 6455 upgrade probe lets the Settings test exercise the
 * same Cloudflare Access leg as the renderer without putting the service token
 * in the URL or adding a second WebSocket dependency.
 */
function probeGatewayWebSocketUpgrade(wsUrl: string, options: ProbeOptions = {}): Promise<ProbeResult> {
  const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
  const readyGraceMs = options.readyGraceMs ?? DEFAULT_READY_GRACE_MS

  let parsed: URL

  try {
    parsed = new URL(wsUrl)
  } catch (error) {
    return Promise.resolve({ ok: false, reason: error instanceof Error ? error.message : String(error) })
  }

  if (parsed.protocol !== 'ws:' && parsed.protocol !== 'wss:') {
    return Promise.resolve({ ok: false, reason: `Unsupported WebSocket protocol: ${parsed.protocol}` })
  }

  return new Promise(resolve => {
    const key = crypto.randomBytes(16).toString('base64')
    const expectedAccept = crypto.createHash('sha1').update(`${key}${WEBSOCKET_GUID}`).digest('base64')
    const client = parsed.protocol === 'wss:' ? https : http
    let settled = false
    let socket: Socket | null = null
    let graceTimer: ReturnType<typeof setTimeout> | null = null

    const finish = (result: ProbeResult) => {
      if (settled) {
        return
      }

      settled = true

      if (graceTimer) {
        clearTimeout(graceTimer)
      }

      try {
        socket?.destroy()
      } catch {
        // Best-effort teardown.
      }

      resolve(result)
    }

    const request = client.request({
      hostname: parsed.hostname,
      method: 'GET',
      path: `${parsed.pathname}${parsed.search}`,
      port: parsed.port || undefined,
      headers: {
        Connection: 'Upgrade',
        Host: parsed.host,
        'Sec-WebSocket-Key': key,
        'Sec-WebSocket-Version': '13',
        Upgrade: 'websocket',
        ...options.headers
      }
    })

    request.on('upgrade', (response, upgradedSocket, head) => {
      socket = upgradedSocket
      const actualAccept = String(response.headers['sec-websocket-accept'] || '')

      if (response.statusCode !== 101 || actualAccept !== expectedAccept) {
        finish({ ok: false, reason: 'The server returned an invalid WebSocket upgrade response.' })

        return
      }

      const onData = (chunk: Buffer) => {
        // A close frame during the grace window is an auth rejection, not a
        // successful gateway.ready frame.
        if (chunk.length > 0 && (chunk[0] & 0x0f) === 0x08) {
          finish({ ok: false, reason: 'The gateway accepted the upgrade and immediately closed the WebSocket.' })

          return
        }

        finish({ ok: true })
      }

      upgradedSocket.once('data', onData)
      upgradedSocket.once('close', () =>
        finish({ ok: false, reason: 'The gateway accepted the upgrade and immediately closed the WebSocket.' })
      )
      upgradedSocket.once('error', error => finish({ ok: false, reason: error.message }))

      if (head.length > 0) {
        onData(head)

        return
      }

      graceTimer = setTimeout(() => finish({ ok: true }), readyGraceMs)
    })

    request.on('response', response => {
      response.resume()
      finish({
        ok: false,
        reason: `WebSocket upgrade returned HTTP ${response.statusCode || 'unknown'}${response.statusMessage ? ` ${response.statusMessage}` : ''}.`
      })
    })
    request.on('error', error => finish({ ok: false, reason: error.message }))

    if (connectTimeoutMs > 0) {
      request.setTimeout(connectTimeoutMs, () => {
        request.destroy()
        finish({ ok: false, reason: `Timed out after ${connectTimeoutMs}ms waiting for the WebSocket upgrade.` })
      })
    }

    request.end()
  })
}

export { probeGatewayWebSocketUpgrade }
