import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createLunarCityPerfPreload } from './lunar-city-perf-preload'

const handshake = {
  bridgeVersion: 1,
  buildSha: 'a'.repeat(40),
  buildStamp: {
    builtAt: '2026-08-31T12:00:00.000Z',
    commit: 'a'.repeat(40),
    dirty: false,
    schemaVersion: 1,
    source: 'local'
  },
  launchNonce: 'nonce-0123456789abcdef-unique',
  mainPid: 41,
  packaged: true,
  processMetricsSource: 'electron.app.getAppMetrics',
  rendererIdentity: { pid: 82, startedAtMs: 1234 },
  supportedPhases: ['baseline-shell', 'mounted-city']
}

function fakeIpc(bootstrap: unknown) {
  const listeners = new Map<string, (...args: unknown[]) => void>()
  const sent: Array<{ channel: string; values: unknown[] }> = []

  return {
    invoke: async (channel: string, ...values: unknown[]) => ({ channel, values }),
    listeners,
    removeListener: (channel: string, listener: (...args: unknown[]) => void) => {
      if (listeners.get(channel) === listener) {
        listeners.delete(channel)
      }
    },
    send: (channel: string, ...values: unknown[]) => sent.push({ channel, values }),
    sendSync: (channel: string) => (channel === 'hermes:lunar-city-perf:bootstrap' ? bootstrap : true),
    sent,
    on: (channel: string, listener: (...args: unknown[]) => void) => listeners.set(channel, listener)
  }
}

function fakeRuntimePort() {
  const posted: unknown[] = []

  const port = {
    closed: false,
    close: () => {
      port.closed = true
    },
    emit: (data: unknown) => port.onmessage?.({ data }),
    onmessage: null as ((event: { data: unknown }) => void) | null,
    postMessage: (value: unknown) => posted.push(value),
    posted,
    start: () => undefined
  }

  return port
}

test('preload bridge is wholly absent when main does not authorize launch', () => {
  const port = fakeRuntimePort()
  assert.equal(createLunarCityPerfPreload(fakeIpc(undefined), port), undefined)
})

test('claims the isolated responder before readiness and exposes no public registration capability', async () => {
  const ipc = fakeIpc(handshake)
  const port = fakeRuntimePort()
  const bridge = createLunarCityPerfPreload(ipc, port)!
  let becameReady = false
  void bridge.ready.then(() => {
    becameReady = true
  })

  assert.equal('renderer' in bridge, false)
  assert.equal('onRequest' in bridge.surface, false)
  assert.equal(becameReady, false)
  assert.equal(ipc.listeners.has('hermes:lunar-city-perf:request'), true)

  port.emit({ type: 'ready' })
  await bridge.ready
  assert.equal(becameReady, true)
})

test('handshake rejects the wrong nonce or bridge version and returns an immutable copy', () => {
  const bridge = createLunarCityPerfPreload(fakeIpc(handshake), fakeRuntimePort())!

  assert.equal(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: 'wrong' }), null)
  assert.equal(bridge.surface.handshake({ bridgeVersion: 2, launchNonce: 'nonce-0123456789abcdef-unique' }), null)

  const first = bridge.surface.handshake({ bridgeVersion: 1, launchNonce: 'nonce-0123456789abcdef-unique' })!
  first.rendererIdentity.pid = 999
  assert.equal(handshake.rendererIdentity.pid, 82)
  assert.equal(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: handshake.launchNonce }), null)
})

test('all capabilities stay blocked until the exact one-shot handshake succeeds', async () => {
  const bridge = createLunarCityPerfPreload(fakeIpc(handshake), fakeRuntimePort())!

  assert.throws(() => bridge.surface.snapshot(), /handshake/u)
  assert.throws(() => bridge.surface.processMetrics(), /handshake/u)
  assert.equal(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: 'wrong' }), null)
  assert.ok(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: handshake.launchNonce }))
  assert.equal(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: handshake.launchNonce }), null)
})

test('surface exposes only bounded read and scenario operations', async () => {
  const bridge = createLunarCityPerfPreload(fakeIpc(handshake), fakeRuntimePort())!
  assert.ok(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: handshake.launchNonce }))

  assert.deepEqual(await bridge.surface.processMetrics(), {
    channel: 'hermes:lunar-city-perf:process-metrics',
    values: []
  })
  assert.deepEqual(await bridge.surface.snapshot(), {
    channel: 'hermes:lunar-city-perf:renderer-request',
    values: ['snapshot', undefined]
  })
  assert.deepEqual(await bridge.surface.prepareBaselineShell(), {
    channel: 'hermes:lunar-city-perf:renderer-request',
    values: ['prepare-baseline-shell', undefined]
  })
  assert.deepEqual(await bridge.surface.mountCity(), {
    channel: 'hermes:lunar-city-perf:renderer-request',
    values: ['mount-city', undefined]
  })
  assert.deepEqual(await bridge.surface.runAction('orbit', { delta: 1 }), {
    channel: 'hermes:lunar-city-perf:renderer-request',
    values: ['scenario-action', { action: 'orbit', payload: { delta: 1 } }]
  })
})

test('isolated runtime port replies once with the original bound request identity', async () => {
  const ipc = fakeIpc(handshake)
  const port = fakeRuntimePort()
  const bridge = createLunarCityPerfPreload(ipc, port)!
  port.emit({ type: 'ready' })
  await bridge.ready
  assert.ok(bridge.surface.handshake({ bridgeVersion: 1, launchNonce: handshake.launchNonce }))
  const listener = ipc.listeners.get('hermes:lunar-city-perf:request')!

  const identity = {
    bridgeVersion: 1,
    buildSha: handshake.buildSha,
    frameId: 3,
    launchNonce: handshake.launchNonce,
    mainPid: handshake.mainPid,
    rendererGeneration: 1,
    rendererPid: handshake.rendererIdentity.pid,
    rendererStartedAtMs: handshake.rendererIdentity.startedAtMs,
    senderId: 7
  }

  listener({}, { action: 'snapshot', identity, payload: { exact: true }, requestId: '7:82:1:1' })
  assert.deepEqual(port.posted, [
    { action: 'snapshot', payload: { exact: true }, requestId: '7:82:1:1', type: 'request' }
  ])
  port.emit({
    requestId: '7:82:1:1',
    type: 'response',
    value: { action: 'snapshot', payload: { exact: true }, proof: 1 }
  })
  assert.deepEqual(ipc.sent, [
    {
      channel: 'hermes:lunar-city-perf:response',
      values: [
        {
          action: 'snapshot',
          identity,
          requestId: '7:82:1:1',
          value: {
            action: 'snapshot',
            payload: { exact: true },
            proof: 1,
            rendererGeneration: 1,
            rendererPid: 82,
            rendererStartedAtMs: 1234
          }
        }
      ]
    }
  ])

  port.emit({ requestId: '7:82:1:1', type: 'response', value: { forged: true } })
  assert.equal(ipc.sent.length, 1)
})
