import assert from 'node:assert/strict'
import { createServer } from 'node:http'

import { test } from 'vitest'

import { resolveGatewayVersion } from './gateway-version'

test('version follows the gateway response across runtime changes', async () => {
  let version = '1.2.3'

  const server = createServer((_request, response) => {
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify({ ok: true, version }))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const baseUrl = `http://127.0.0.1:${address.port}`

  const request = async (endpoint: string): Promise<unknown> => {
    assert.equal(endpoint, '/api/health')

    return (await fetch(`${baseUrl}${endpoint}`)).json()
  }

  try {
    assert.equal(await resolveGatewayVersion(request), version)
    version = '4.5.6'
    assert.equal(await resolveGatewayVersion(request), version)
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})

test('unavailable gateway version stays unknown instead of using another install', async () => {
  for (const response of [null, {}, { version: 42 }, { version: '' }]) {
    assert.equal(await resolveGatewayVersion(async () => response), '')
  }

  assert.equal(await resolveGatewayVersion(async () => { throw new Error('offline') }), '')
})
