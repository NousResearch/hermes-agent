import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createPoolRetirementClient } from './pool-retire-http'

const DESCRIPTOR = { baseUrl: 'http://127.0.0.1:53412', mode: 'remote', token: 'descriptor-token' }

test('a local child is fenced at its own loopback port', async () => {
  const calls: { token: null | string; url: string }[] = []
  const client = createPoolRetirementClient(async (url, token) => {
    calls.push({ url, token })

    return { ok: true, idle: true, token: 'permit' }
  })

  assert.equal(await client.prepare('local', { process: {}, port: 51234, token: 'child-token' }), 'permit')
  assert.deepEqual(calls, [{ url: 'http://127.0.0.1:51234/api/health/retirement', token: 'child-token' }])
  // A child with no announced port has no reachable fence — never authority.
  assert.equal(await client.prepare('local', { process: {}, port: null, token: 'child-token' }), null)
})

test('a pooled descriptor is fenced through its resolved descriptor, never a raw URL', async () => {
  const urls: string[] = []
  const descriptorCalls: { path: string; token: null | string | undefined }[] = []
  const client = createPoolRetirementClient(
    async url => { urls.push(url); return { ok: false } },
    async (descriptor, path, options) => {
      descriptorCalls.push({ path, token: descriptor.token })
      assert.equal(options.method, 'POST')
      assert.equal((options.body as { action: string }).action, 'prepare')

      return { ok: true, idle: true, token: 'remote-permit' }
    }
  )

  assert.equal(await client.prepare('remote', { process: null, connectionPromise: Promise.resolve(DESCRIPTOR) }), 'remote-permit')
  assert.deepEqual(urls, [], 'a descriptor entry must not be dialled as a bare loopback URL')
  assert.deepEqual(descriptorCalls, [{ path: '/api/health/retirement', token: 'descriptor-token' }])
})

test('a descriptor entry without a descriptor transport or connection grants no authority', async () => {
  const withoutTransport = createPoolRetirementClient(async () => ({ ok: true, idle: true, token: 'permit' }))

  assert.equal(await withoutTransport.prepare('remote', { process: null, connectionPromise: Promise.resolve(DESCRIPTOR) }), null)
  assert.equal(await withoutTransport.commit('remote', { process: null, connectionPromise: Promise.resolve(DESCRIPTOR) }, 'permit'), false)

  const withTransport = createPoolRetirementClient(
    async () => ({ ok: true, idle: true, token: 'permit' }),
    async () => ({ ok: true, idle: true, token: 'remote-permit' })
  )

  assert.equal(await withTransport.prepare('remote', { process: null, connectionPromise: null }), null)
  // An unreachable backend cannot produce a permit: the fence is never assumed.
  const unreachable = Promise.reject(new Error('tunnel down'))
  unreachable.catch(() => undefined)
  assert.equal(await withTransport.prepare('remote', { process: null, connectionPromise: unreachable }), null)
  assert.equal(await withTransport.prepare('remote', { process: null, connectionPromise: Promise.resolve({ baseUrl: '' }) }), null)
  assert.equal(await withTransport.prepare('remote', { process: null, connectionPromise: Promise.resolve(DESCRIPTOR) }), 'remote-permit')
})

test('a connect that never settles grants no authority instead of holding the arbiter', async () => {
  const descriptorCalls: string[] = []
  const client = createPoolRetirementClient(
    async () => ({ ok: false }),
    async () => {
      descriptorCalls.push('transport')

      return { ok: true, idle: true, token: 'remote-permit' }
    },
    25
  )
  const hung = new Promise<never>(() => {})

  assert.equal(await client.prepare('remote', { process: null, connectionPromise: hung }), null)
  assert.deepEqual(descriptorCalls, [], 'a connect that never resolves is never fenced')
})

test('busy or indeterminate replies never mint a permit', async () => {
  for (const reply of [{ ok: false, idle: false }, { ok: false, idle: null }, { ok: true, idle: false }, { ok: true, idle: true, token: '' }]) {
    const client = createPoolRetirementClient(async () => reply)

    assert.equal(await client.prepare('local', { process: {}, port: 51234, token: 'child-token' }), null, JSON.stringify(reply))
  }
})
