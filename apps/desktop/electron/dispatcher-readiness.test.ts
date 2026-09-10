import assert from 'node:assert/strict'

import { test } from 'vitest'

import { DispatcherReadinessError, ensureKanbanDispatcherReady, runDispatcherReadinessGate } from './dispatcher-readiness'

test('runDispatcherReadinessGate advances the boot phase before checking readiness', async () => {
  const events: string[] = []

  await runDispatcherReadinessGate(
    'http://127.0.0.1:9000',
    'session-token',
    async () => {
      events.push('readiness-checked')
      return { status: 'ready', ready: true, gateway_pid: 1, message: 'ok' }
    },
    async (id: string) => {
      events.push(`phase:${id}`)
    }
  )

  assert.deepEqual(
    events,
    ['phase:backend.dispatcher', 'readiness-checked'],
    'boot phase must be advanced before the readiness endpoint is queried'
  )
})

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

test('allows Desktop startup when the embedded dispatcher is disabled', async () => {
  const calls: string[] = []

  const result = await ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async url => {
    calls.push(url)
    return { status: 'disabled', ready: false, gateway_pid: null, message: 'dispatcher is disabled' }
  })

  assert.equal(result.status, 'disabled')
  assert.equal(result.ready, false)
  assert.deepEqual(calls, ['http://127.0.0.1:9000/api/plugins/kanban/dispatcher-readiness'])
})

test('allows Desktop startup when the Kanban plugin is absent (404 — not mounted)', async () => {
  // When the Kanban plugin is explicitly disabled or removed, its API router is
  // never mounted, so the readiness endpoint returns 404. Desktop must not treat
  // this as a startup failure; it should behave identically to { status: "disabled" }.
  const result = await ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async () => {
    throw new Error('404: Not Found')
  })

  assert.equal(result.status, 'disabled')
  assert.equal(result.ready, false)
})

test('blocks startup without starting a gateway for unknown dispatcher state', async () => {
  const calls: string[] = []

  await assert.rejects(
    ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async url => {
      calls.push(url)
      return { status: 'unknown', ready: false, gateway_pid: null, message: 'dispatcher is unknown' }
    }),
    error => {
      assert.ok(error instanceof DispatcherReadinessError)
      assert.equal(error.code, 'dispatcher-offline')
      assert.equal(error.blocking, true)
      assert.match(error.message, /KANBAN_DISPATCHER_OFFLINE/)
      assert.match(error.message, /unknown/)

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
