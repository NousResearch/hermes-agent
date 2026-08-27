import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { DispatcherReadinessError, ensureKanbanDispatcherReady } from './dispatcher-readiness'

const mainSource = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), 'main.ts'), 'utf8')

test('local Desktop startup gates backend.ready on dispatcher readiness', () => {
  const readinessCall = mainSource.indexOf('await ensureKanbanDispatcherReady(baseUrl, authToken, fetchJson)')
  const readyPhase = mainSource.indexOf("phase: 'backend.ready'", readinessCall)

  assert.ok(readinessCall > 0, 'local startup must invoke the dispatcher readiness gate')
  assert.ok(readyPhase > readinessCall, 'backend.ready must be published only after dispatcher readiness')
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

test.each(['offline', 'disabled', 'unknown'])('blocks startup for %s dispatcher state', async status => {
  await assert.rejects(
    ensureKanbanDispatcherReady('http://127.0.0.1:9000', 'session-token', async () => ({
      status,
      ready: false,
      gateway_pid: null,
      message: `dispatcher is ${status}`
    })),
    error => {
      assert.ok(error instanceof DispatcherReadinessError)
      assert.equal(error.code, 'dispatcher-offline')
      assert.equal(error.blocking, true)
      assert.match(error.message, /KANBAN_DISPATCHER_OFFLINE/)
      assert.match(error.message, new RegExp(status))

      return true
    }
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
