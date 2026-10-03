import assert from 'node:assert/strict'

import { test } from 'vitest'

import { attachToHostBackend } from './host-backend-attach'

const LEDGER = JSON.stringify([
  {
    argv: 'hermes serve --host 127.0.0.1 --port 0',
    create_time: 1000,
    host: '127.0.0.1',
    install: 'abc',
    pid: 4711,
    port: 65238,
    profile: 'ops',
    purpose: 'serve',
    registered_at: 2000
  }
])

function baseDeps(overrides = {}) {
  return {
    log: () => {},
    probeWebSocket: async () => ({ ok: true }),
    readLedger: () => LEDGER,
    resolveServedToken: async () => 'served-token',
    waitForReady: async () => undefined,
    ...overrides
  }
}

test('a transient WS timeout retries once while the pid is alive instead of spawning', async () => {
  let probes = 0

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    baseDeps({
      isPidAlive: () => true,
      probeWebSocket: async () => {
        probes += 1

        if (probes === 1) {
          return { ok: false, reason: 'Timed out after 10000ms waiting for the WebSocket to open.' }
        }

        return { ok: true }
      }
    })
  )

  assert.equal(probes, 2)
  assert.equal(attached?.pid, 4711)
})

test('a transient WS timeout with a dead pid does not retry', async () => {
  let probes = 0

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    baseDeps({
      isPidAlive: () => false,
      probeWebSocket: async () => {
        probes += 1

        return { ok: false, reason: 'Timed out after 10000ms waiting for the WebSocket to open.' }
      }
    })
  )

  // Dead pids are skipped before any network I/O, so the probe never runs.
  assert.equal(probes, 0)
  assert.equal(attached, null)
})

test('an auth rejection never retries even while the pid is alive', async () => {
  let probes = 0

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    baseDeps({
      isPidAlive: () => true,
      probeWebSocket: async () => {
        probes += 1

        return { ok: false, reason: 'unauthorized' }
      }
    })
  )

  assert.equal(probes, 1)
  assert.equal(attached, null)
})

test('a persistent transient WS timeout exhausts the single retry and refuses the record', async () => {
  let probes = 0

  const attached = await attachToHostBackend(
    { isolated: false, ledgerPath: '/ledger.json' },
    baseDeps({
      isPidAlive: () => true,
      probeWebSocket: async () => {
        probes += 1

        return { ok: false, reason: 'Timed out after 10000ms waiting for the WebSocket to open.' }
      }
    })
  )

  assert.equal(probes, 2)
  assert.equal(attached, null)
})
