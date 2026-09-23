import assert from 'node:assert/strict'
import { createServer, type RequestListener, type Server } from 'node:http'

import { afterEach, test } from 'vitest'

import { createGatewayJsonRuntime } from './gateway-json-runtime'

const servers: Server[] = []

async function loopback(onRequest: RequestListener) {
  const server = createServer(onRequest)
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  return `http://127.0.0.1:${address.port}`
}

afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => server.close(() => resolve()))))
})

test('public probe remains credential-free while token and bearer JSON calls use live remote headers', async () => {
  const seen: Array<{ body: string; headers: Record<string, string | string[] | undefined> }> = []

  const baseUrl = await loopback((request, response) => {
    const chunks: Buffer[] = []
    request.on('data', chunk => chunks.push(Buffer.from(chunk)))
    request.on('end', () => {
      seen.push({ body: Buffer.concat(chunks).toString('utf8'), headers: request.headers })
      response.writeHead(200, { 'content-type': 'application/json', connection: 'close' })
      response.end(JSON.stringify({ ok: true }))
    })
  })

  let remoteHeader = 'first'
  const runtime = createGatewayJsonRuntime({ headersForRemoteRequest: () => ({ 'X-Remote-Header': remoteHeader }) })

  assert.deepEqual(await runtime.fetchPublicJson(`${baseUrl}/api/status`), { ok: true })
  remoteHeader = 'second'
  assert.deepEqual(
    await runtime.fetchJson(`${baseUrl}/api/session`, 'secret', { method: 'POST', body: { answer: 42 } }),
    {
      ok: true
    }
  )
  assert.deepEqual(await runtime.fetchJson(`${baseUrl}/api/session`, null, { bearer: 'bearer-secret' }), { ok: true })

  assert.equal(seen.length, 3)
  assert.equal(seen[0].headers['x-hermes-session-token'], undefined)
  assert.equal(seen[0].headers.authorization, undefined)
  assert.equal(seen[0].headers['x-remote-header'], 'first')
  assert.equal(seen[1].headers['x-hermes-session-token'], 'secret')
  assert.equal(seen[1].headers['x-remote-header'], 'second')
  assert.equal(seen[1].body, '{"answer":42}')
  assert.equal(seen[2].headers.authorization, 'Bearer bearer-secret')
})

test('a POST whose body reached the server is not replayed after a lost response', async () => {
  let submissions = 0

  const baseUrl = await loopback((request, _response) => {
    request.resume()
    request.on('end', () => {
      submissions += 1
      request.socket.destroy()
    })
  })

  const runtime = createGatewayJsonRuntime({ headersForRemoteRequest: () => ({}) })

  await assert.rejects(runtime.fetchJson(`${baseUrl}/submit`, 'token', { method: 'POST', body: { submitted: true } }))
  assert.equal(submissions, 1)
})

test('a request timeout after POST submission surfaces without resubmitting', async () => {
  let submissions = 0

  const baseUrl = await loopback((request, _response) => {
    request.resume()
    request.on('end', () => {
      submissions += 1
    })
  })

  const runtime = createGatewayJsonRuntime({ headersForRemoteRequest: () => ({}) })

  await assert.rejects(
    runtime.fetchJson(`${baseUrl}/submit`, 'token', { method: 'POST', body: { submitted: true }, timeoutMs: 25 }),
    /Timed out connecting to Hermes backend after 25ms/
  )
  assert.equal(submissions, 1)
})
