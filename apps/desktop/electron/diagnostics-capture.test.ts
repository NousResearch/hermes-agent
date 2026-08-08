import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  classifyTransportError,
  createCaptureController,
  createMainRing,
  routePrefix,
  sanitizeAppMetrics
} from './diagnostics-capture'

// A minimal fake BrowserWindow: records everything main sent it and lets a test
// fire the webContents responsiveness edges, mirroring the slice of the
// Electron API the controller actually touches (see session-windows.test.ts).
function makeFakeWindow(id: number) {
  const sent: { channel: string; payload: any }[] = []
  const contentsListeners: Record<string, () => void> = {}
  let destroyed = false

  return {
    id,
    sent,
    isDestroyed: () => destroyed,
    destroy() {
      destroyed = true
    },
    on() {},
    emitContents(event: string) {
      contentsListeners[event]?.()
    },
    collectRequests: () => sent.filter(entry => entry.channel === 'diagnostics:collect'),
    webContents: {
      isDestroyed: () => destroyed,
      send(channel: string, payload: any) {
        sent.push({ channel, payload })
      },
      on(event: string, listener: () => void) {
        contentsListeners[event] = listener
      }
    }
  }
}

// Hand-driven timers: nothing fires until the test says so, so "sampled only
// while armed" is an assertion about registration AND about firing.
function makeFakeTimers() {
  const intervals = new Map<number, { fn: () => void; ms: number }>()
  const timeouts = new Map<number, { fn: () => void; ms: number }>()
  let nextHandle = 1

  return {
    intervals,
    timeouts,
    tickIntervals(times = 1) {
      for (let i = 0; i < times; i += 1) {
        for (const entry of [...intervals.values()]) {
          entry.fn()
        }
      }
    },
    fireTimeouts() {
      for (const entry of [...timeouts.values()]) {
        entry.fn()
      }
    },
    timers: {
      setInterval(fn: () => void, ms: number) {
        const handle = nextHandle++
        intervals.set(handle, { fn, ms })

        return handle
      },
      clearInterval(handle: any) {
        intervals.delete(handle)
      },
      setTimeout(fn: () => void, ms: number) {
        const handle = nextHandle++
        timeouts.set(handle, { fn, ms })

        return handle
      },
      clearTimeout(handle: any) {
        timeouts.delete(handle)
      }
    }
  }
}

function makeFakeGateway(overrides: any = {}) {
  const calls: string[] = []

  return {
    calls,
    client: {
      async arm(input) {
        calls.push('arm')

        return overrides.arm ? overrides.arm(input) : { ok: true, monotonicAnchorMs: 500 }
      },
      async collect(captureId) {
        calls.push('collect')

        return overrides.collect
          ? overrides.collect(captureId)
          : { ok: true, stream: { captureId, monotonicAnchorMs: 500, events: [], dropped: 0 } }
      },
      async disarm() {
        calls.push('disarm')
      }
    }
  }
}

function makeController(options: any = {}) {
  const timers = makeFakeTimers()
  const windows = options.windows ?? []
  let clock = 1000

  const controller = createCaptureController({
    now: () => clock,
    wallClock: () => 1_700_000_000_000,
    randomUUID: () => 'cap-test',
    listWindows: () => windows,
    timers: timers.timers,
    ...options.controller
  })

  return {
    controller,
    timers,
    windows,
    advance(ms: number) {
      clock += ms
    },
    setClock(value: number) {
      clock = value
    }
  }
}

/** Answer every outstanding collect request for a window with `snapshot`. */
function answerCollect(controller: any, win: any, snapshot: any) {
  for (const request of win.collectRequests()) {
    controller.handleCollectResult({ requestId: request.payload.requestId, snapshot })
  }
}

const rendererSnapshot = {
  captureId: 'cap-test',
  wallClockAnchorMs: 1_700_000_000_000,
  monotonicAnchorMs: 10,
  events: [{ type: 'long_frame', t: 42 }],
  droppedEvents: 2
}

test('arming pushes the capture id and wall-clock anchor to every window', async () => {
  const a = makeFakeWindow(1)
  const b = makeFakeWindow(2)
  const harness = makeController({ windows: [a, b] })

  const started = await harness.controller.start()

  assert.equal(started.captureId, 'cap-test')
  assert.equal(started.wallClockAnchorMs, 1_700_000_000_000)

  for (const win of [a, b]) {
    assert.deepEqual(win.sent, [
      { channel: 'diagnostics:arm', payload: { captureId: 'cap-test', wallClockAnchorMs: 1_700_000_000_000 } }
    ])
  }

  assert.equal(harness.controller.isArmed(), true)
})

test('a window opened mid-capture is armed as it attaches', async () => {
  const a = makeFakeWindow(1)
  const harness = makeController({ windows: [a] })
  await harness.controller.start()

  const late = makeFakeWindow(2)
  harness.controller.attachWindow(late as never)

  assert.deepEqual(late.sent[0], {
    channel: 'diagnostics:arm',
    payload: { captureId: 'cap-test', wallClockAnchorMs: 1_700_000_000_000 }
  })
})

test('stopping disarms every window and clears the armed state', async () => {
  const a = makeFakeWindow(1)
  const harness = makeController({ windows: [a] })
  await harness.controller.start()

  const stopping = harness.controller.stop()
  answerCollect(harness.controller, a, rendererSnapshot)
  const bundle = await stopping

  assert.equal(harness.controller.isArmed(), false)
  assert.equal(harness.controller.captureId(), null)
  assert.equal(a.sent.at(-1)?.channel, 'diagnostics:disarm')
  assert.deepEqual(bundle?.renderer, [{ windowId: 1, ...rendererSnapshot }])
  assert.deepEqual(bundle?.anchors, { wallClockAnchorMs: 1_700_000_000_000, mainMonotonicAnchorMs: 1000 })
})

test('a renderer that never answers is skipped instead of hanging the export', async () => {
  const a = makeFakeWindow(1)
  const b = makeFakeWindow(2)
  const harness = makeController({ windows: [a, b] })
  await harness.controller.start()

  const stopping = harness.controller.stop()
  answerCollect(harness.controller, a, rendererSnapshot)
  // b is wedged: fire its collect timeout instead of replying.
  harness.timers.fireTimeouts()
  const bundle = await stopping

  assert.deepEqual(
    bundle?.renderer.map(stream => stream.windowId),
    [1]
  )
})

test('app metrics are sampled only while armed', async () => {
  let samples = 0

  const harness = makeController({
    windows: [],
    controller: {
      getAppMetrics: () => {
        samples += 1

        return [{ pid: 10, type: 'Browser', cpu: { percentCPUUsage: 1 }, memory: { workingSetSize: 2048 } }]
      }
    }
  })

  // Disarmed: no interval exists at all, so nothing can sample.
  assert.equal(harness.timers.intervals.size, 0)
  harness.timers.tickIntervals()
  assert.equal(samples, 0)

  await harness.controller.start()
  assert.equal(harness.timers.intervals.size, 1)
  harness.timers.tickIntervals(2)

  const stopping = harness.controller.stop()
  const bundle = await stopping

  // Two ticks plus the closing sample taken by stop().
  assert.equal(samples, 3)
  assert.equal(harness.timers.intervals.size, 0)
  assert.equal(bundle?.main.filter(event => event.type === 'app_metrics').length, 3)

  harness.timers.tickIntervals()
  assert.equal(samples, 3)
})

test('window responsiveness edges are recorded with the blocked duration', async () => {
  const a = makeFakeWindow(1)
  const harness = makeController({ windows: [a] })
  await harness.controller.start()

  a.emitContents('unresponsive')
  harness.advance(2500)
  a.emitContents('responsive')

  const stopping = harness.controller.stop()
  answerCollect(harness.controller, a, rendererSnapshot)
  const bundle = await stopping

  assert.deepEqual(bundle?.main, [
    { type: 'window_unresponsive', t: 1000, windowId: 1 },
    { type: 'window_responsive', t: 3500, windowId: 1, blockedMs: 2500 }
  ])
})

test('responsiveness edges outside a capture are dropped', async () => {
  const a = makeFakeWindow(1)
  const harness = makeController({ windows: [a] })

  harness.controller.attachWindow(a as never)
  a.emitContents('unresponsive')
  a.emitContents('responsive')

  await harness.controller.start()
  const stopping = harness.controller.stop()
  answerCollect(harness.controller, a, rendererSnapshot)
  const bundle = await stopping

  assert.deepEqual(bundle?.main, [])
})

test('renderer transport failures become sanitized transport events', async () => {
  const harness = makeController({ windows: [] })

  // Disarmed failures cost nothing and are not retained.
  harness.controller.recordTransportError({
    channel: 'hermes:api',
    path: '/api/sessions/abc/messages',
    durationMs: 10,
    error: new Error('boom')
  })

  await harness.controller.start()
  harness.controller.recordTransportError({
    channel: 'hermes:api',
    path: '/api/sessions/abc-123/messages?limit=50',
    durationMs: 60_004,
    error: new Error('Timed out connecting to Hermes backend after 60000ms')
  })
  harness.controller.recordTransportError({
    channel: 'hermes:api',
    path: '/api/status',
    durationMs: 12,
    error: new Error('502: bad gateway')
  })

  const bundle = await harness.controller.stop()
  const transport = bundle?.main.filter(event => event.type === 'transport_error')

  assert.deepEqual(transport, [
    {
      type: 'transport_error',
      t: 1000,
      channel: 'hermes:api',
      route: '/api/sessions',
      durationMs: 60_004,
      errorClass: 'timeout'
    },
    {
      type: 'transport_error',
      t: 1000,
      channel: 'hermes:api',
      route: '/api/status',
      durationMs: 12,
      errorClass: 'http_502'
    }
  ])
})

test('a remote gateway is never armed and its stream is marked absent', async () => {
  const gateway = makeFakeGateway()

  const harness = makeController({
    windows: [],
    controller: { gateway: gateway.client, isRemoteGateway: () => true }
  })

  await harness.controller.start()
  const bundle = await harness.controller.stop()

  assert.deepEqual(bundle?.gateway, { absent: 'remote-gateway' })
  assert.deepEqual(gateway.calls, [])
})

test('a local gateway ring is pulled into the bundle', async () => {
  const gateway = makeFakeGateway({
    collect: captureId => ({
      ok: true,
      stream: { captureId, monotonicAnchorMs: 500, events: [{ type: 'loop_stall', t: 7 }], dropped: 1 }
    })
  })

  const harness = makeController({
    windows: [],
    controller: { gateway: gateway.client, isRemoteGateway: () => false }
  })

  await harness.controller.start()
  const bundle = await harness.controller.stop()

  assert.deepEqual(bundle?.gateway, {
    captureId: 'cap-test',
    monotonicAnchorMs: 500,
    events: [{ type: 'loop_stall', t: 7 }],
    dropped: 1
  })
  assert.deepEqual(gateway.calls, ['arm', 'collect', 'disarm'])
})

test('a gateway that refuses to arm is marked absent and never pulled', async () => {
  const gateway = makeFakeGateway({ arm: () => ({ ok: false, reason: 'unauthenticated' }) })

  const harness = makeController({
    windows: [],
    controller: { gateway: gateway.client, isRemoteGateway: () => false }
  })

  await harness.controller.start()
  const bundle = await harness.controller.stop()

  assert.deepEqual(bundle?.gateway, { absent: 'unauthenticated' })
  assert.equal(gateway.calls.includes('collect'), false)
})

test('no gateway client at all still produces a complete bundle', async () => {
  const harness = makeController({ windows: [], controller: { gateway: null } })

  await harness.controller.start()
  const bundle = await harness.controller.stop()

  assert.deepEqual(bundle?.gateway, { absent: 'disabled' })
  assert.equal(bundle?.captureId, 'cap-test')
})

test('the completed bundle stays readable for the exporter', async () => {
  const harness = makeController({ windows: [] })

  assert.equal(harness.controller.lastCapture(), null)
  await harness.controller.start()
  const bundle = await harness.controller.stop()

  assert.equal(harness.controller.lastCapture(), bundle)
  assert.equal(await harness.controller.stop(), null)
})

test('the main ring is bounded by count and by age', () => {
  const ring = createMainRing({ maxEvents: 3, maxAgeMs: 1000 })

  for (let i = 0; i < 5; i += 1) {
    ring.push({ type: 'app_metrics', t: i, processes: [] } as never)
  }

  assert.equal(ring.size, 3)
  assert.equal(ring.dropped, 2)
  assert.deepEqual(
    ring.entries().map(event => event.t),
    [2, 3, 4]
  )

  ring.push({ type: 'app_metrics', t: 9000, processes: [] } as never)

  assert.deepEqual(
    ring.entries().map(event => event.t),
    [9000]
  )
  assert.equal(ring.dropped, 5)
})

test('metrics samples keep numbers only', () => {
  assert.deepEqual(
    sanitizeAppMetrics([
      {
        pid: 42,
        type: 'Utility',
        name: 'Network Service',
        cpu: { percentCPUUsage: 1.23456 },
        memory: { workingSetSize: 8192 }
      } as never
    ]),
    [{ pid: 42, type: 'Utility', percentCpu: 1.23, workingSetKb: 8192 }]
  )
  assert.deepEqual(sanitizeAppMetrics(undefined), [])
})

test('route prefixes drop ids, queries and fragments', () => {
  assert.equal(routePrefix('/api/sessions/abc-123/messages?limit=50#x'), '/api/sessions')
  assert.equal(routePrefix('/api/status'), '/api/status')
  assert.equal(routePrefix('/'), '/')
  assert.equal(routePrefix(undefined), null)
})

test('transport errors are classified without keeping their message', () => {
  assert.equal(classifyTransportError(new Error('Timed out connecting to Hermes backend after 60000ms')), 'timeout')
  assert.equal(classifyTransportError(new Error('404: not found')), 'http_404')
  assert.equal(classifyTransportError(Object.assign(new Error('x'), { code: 'ECONNREFUSED' })), 'connect_ECONNREFUSED')
  assert.equal(classifyTransportError(Object.assign(new Error('x'), { code: 'ETIMEDOUT' })), 'timeout')
  assert.equal(classifyTransportError(new Error('something else')), 'error')
})
