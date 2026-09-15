import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveGatewayVersion } from './gateway-version'

test('version follows each supplied gateway response without caching', async (): Promise<void> => {
  let version: string = '1.2.3'

  const request: (endpoint: string) => Promise<{ version: string }> = async (endpoint: string): Promise<{ version: string }> => {
    assert.equal(endpoint, '/api/health')

    return { version }
  }

  assert.equal(await resolveGatewayVersion(request), '1.2.3')
  version = '4.5.6'
  assert.equal(await resolveGatewayVersion(request), '4.5.6')
})

test('unavailable gateway version stays unknown instead of using another install', async () => {
  for (const response of [null, {}, { version: 42 }, { version: '' }]) {
    assert.equal(await resolveGatewayVersion(async () => response), '')
  }

  assert.equal(await resolveGatewayVersion(async () => { throw new Error('offline') }), '')
})
