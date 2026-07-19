import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import http from 'node:http'

import { test } from 'vitest'

import { probeGatewayWebSocketUpgrade } from './gateway-ws-upgrade-probe'

const WEBSOCKET_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

test('custom-header WebSocket probe sends the Cloudflare pair and validates the upgrade', async () => {
  let receivedId = ''
  let receivedSecret = ''
  const server = http.createServer()

  server.on('upgrade', (request, socket) => {
    receivedId = String(request.headers['cf-access-client-id'] || '')
    receivedSecret = String(request.headers['cf-access-client-secret'] || '')

    const key = String(request.headers['sec-websocket-key'] || '')
    const accept = crypto.createHash('sha1').update(`${key}${WEBSOCKET_GUID}`).digest('base64')
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
    )
    // Minimal unmasked text frame ("ok") proves the accepted gateway speaks.
    socket.end(Buffer.from([0x81, 0x02, 0x6f, 0x6b]))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  try {
    const address = server.address()
    assert.ok(address && typeof address === 'object')

    const result = await probeGatewayWebSocketUpgrade(`ws://127.0.0.1:${address.port}/api/ws?token=hermes`, {
      connectTimeoutMs: 1_000,
      readyGraceMs: 20,
      headers: {
        'CF-Access-Client-Id': 'client.access',
        'CF-Access-Client-Secret': 'secret-value'
      }
    })

    assert.deepEqual(result, { ok: true })
    assert.equal(receivedId, 'client.access')
    assert.equal(receivedSecret, 'secret-value')
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('custom-header WebSocket probe reports a rejected HTTP upgrade', async () => {
  const server = http.createServer((_request, response) => {
    response.writeHead(403).end('forbidden')
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  try {
    const address = server.address()
    assert.ok(address && typeof address === 'object')

    const result = await probeGatewayWebSocketUpgrade(`ws://127.0.0.1:${address.port}/api/ws`, {
      connectTimeoutMs: 1_000
    })

    assert.equal(result.ok, false)
    assert.match(result.reason || '', /HTTP 403/)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
