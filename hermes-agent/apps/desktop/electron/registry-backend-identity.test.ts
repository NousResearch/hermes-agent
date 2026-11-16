import assert from 'node:assert/strict'

import { test } from 'vitest'

import { connectionInstallIds } from './connection-caches'
import {
  boundedInstallId,
  createPrimaryInstallIdResolver,
  installIdFromStatus,
  probeCachedConnectionInstallId
} from './registry-backend-identity'

test('installIdFromStatus reads a non-empty install_id only', () => {
  assert.equal(installIdFromStatus({ install_id: ' abc ' }), 'abc')
  assert.equal(installIdFromStatus({ install_id: '' }), undefined)
  assert.equal(installIdFromStatus({}), undefined)
  assert.equal(installIdFromStatus(null), undefined)
})

test('a slow or failing identity probe reads as unknown', async () => {
  assert.equal(await boundedInstallId(new Promise(() => {}), 5), undefined)
  assert.equal(await boundedInstallId(Promise.reject(new Error('down'))), undefined)
  assert.equal(await boundedInstallId(Promise.resolve('abc')), 'abc')
})

test('the primary install id is cached per backend URL and re-probed when the primary moves', async () => {
  let url = 'http://local'
  let probes = 0

  const resolve = createPrimaryInstallIdResolver({
    resolve: async () => ({ baseUrl: url }),
    status: async (descriptor: any) => {
      probes += 1

      return { install_id: descriptor.baseUrl === 'http://local' ? 'L' : 'R' }
    },
    ttlMs: 60_000,
    negativeTtlMs: 1_000
  })

  assert.equal(await resolve(), 'L')
  assert.equal(await resolve(), 'L')
  assert.equal(probes, 1)
  url = 'http://remote'
  assert.equal(await resolve(), 'R')
  assert.equal(probes, 2)
})

test('an unresolvable primary has no identity', async () => {
  const resolve = createPrimaryInstallIdResolver({
    resolve: async () => {
      throw new Error('not started')
    },
    status: async () => ({ install_id: 'x' }),
    ttlMs: 1,
    negativeTtlMs: 1
  })

  assert.equal(await resolve(), undefined)
})

test('a connection install id is cached, and a failed re-probe keeps the known id', async () => {
  connectionInstallIds.clear()
  let calls = 0

  assert.equal(await probeCachedConnectionInstallId('c', async () => (calls++, { install_id: 'abc' })), 'abc')
  assert.equal(await probeCachedConnectionInstallId('c', async () => (calls++, { install_id: 'zzz' })), 'abc')
  assert.equal(calls, 1)
  connectionInstallIds.set('c', { id: 'abc', ts: 0 })
  assert.equal(
    await probeCachedConnectionInstallId('c', async () => {
      throw new Error('down')
    }),
    'abc'
  )
  connectionInstallIds.clear()
})
