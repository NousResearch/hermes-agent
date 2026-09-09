import assert from 'node:assert/strict'

import { test } from 'vitest'

import { DispatcherReadinessError, ensureKanbanDispatcherReady } from './dispatcher-readiness'

test('accepts a live gateway-owned dispatcher', async () => {
  const calls: Array<[string, string | null]> = []

  const result = await ensureKanbanDispatcherReady('http://127.0.0.1:9000/', 'session-token', async (url, token) => {
    calls.push([url, token])

    return { status: 'ready', ready: true, gateway_pid: 4321, message: 'dispatch enabled' }
  })

  assert.equal(result.gateway_pid, 4321)
  assert.deepEqual(calls, [['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness', 'session-token']])
})

test('starts one supervised gateway when the dispatcher is offline and waits for readiness', async () => {
  const calls: Array<[string, string | null, string]> = []
  const responses = [
    { status: 'offline', ready: false, gateway_pid: null, message: 'gateway is offline' },
    { ok: true, pid: 7654, name: 'gateway-start' },
    { status: 'offline', ready: false, gateway_pid: null, message: 'gateway is starting' },
    { status: 'ready', ready: true, gateway_pid: 4321, message: 'dispatch enabled' }
  ]

  const result = await ensureKanbanDispatcherReady(
    'http://127.0.0.1:9000/',
    'session-token',
    async (url, token, options = {}) => {
      calls.push([url, token, options.method || 'GET'])
      return responses.shift()
    },
    { attempts: 2, pollMs: 0, sleep: async () => {} }
  )

  assert.equal(result.gateway_pid, 4321)
  assert.deepEqual(calls, [
    ['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness', 'session-token', 'GET'],
    ['http://127.0.0.1:9000/api/gateway/start', 'session-token', 'POST'],
    ['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness', 'session-token', 'GET'],
    ['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness', 'session-token', 'GET']
  ])
})

test.each(['disabled', 'unknown'])('blocks startup without starting a gateway for %s dispatcher state', async status => {
  const calls: string[] = []

  await assert.rejects(
    ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async url => {
      calls.push(url)
      return {
        status,
        ready: false,
        gateway_pid: null,
        message: `dispatcher is ${status}`
      }
    }),
    error => {
      assert.ok(error instanceof DispatcherReadinessError)
      assert.equal(error.code, 'dispatcher-offline')
      assert.equal(error.blocking, true)
      assert.match(error.message, /KANBAN_DISPATCHER_OFFLINE/)
      assert.match(error.message, new RegExp(status))

      return true
    }
  )

  assert.deepEqual(calls, ['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness'])
})

test('surfaces a supervised gateway start failure', async () => {
  await assert.rejects(
    ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async (url, _token, options = {}) => {
      if (options.method === 'POST') {
        throw new Error('launchctl bootstrap failed')
      }

      return { status: 'offline', ready: false, gateway_pid: null, message: 'gateway is offline' }
    }),
    /KANBAN_DISPATCHER_OFFLINE.*could not start.*launchctl bootstrap failed/
  )
})

test('blocks startup when dispatcher readiness cannot be verified', async () => {
  await assert.rejects(
    ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async () => {
      throw new Error('503 unavailable')
    }),
    /KANBAN_DISPATCHER_OFFLINE.*could not be verified.*503 unavailable/
  )
})
