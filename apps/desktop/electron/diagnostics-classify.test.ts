import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyCapture } from './diagnostics-classify'

// Fixtures are single-signal on purpose: the point of each is that the
// classifier reaches for the RIGHT label, which only means something when the
// other four have nothing to go on.
function bundle(overrides: any = {}) {
  return {
    captureId: 'cap-1',
    anchors: { wallClockAnchorMs: 1_700_000_000_000, mainMonotonicAnchorMs: 0 },
    renderer: [],
    main: [],
    mainDropped: 0,
    gateway: { absent: 'disabled' },
    ...overrides
  } as any
}

function rendererStream(events: any[]) {
  return [{ windowId: 1, captureId: 'cap-1', wallClockAnchorMs: 0, monotonicAnchorMs: 0, droppedEvents: 0, events }]
}

function longFrame(t: number, ms: number) {
  return { type: 'long_frame', t, ms, styleMs: 1, blockingMs: ms - 10, scripts: [] }
}

function loopDrift(t: number, driftS: number) {
  return { kind: 'loop_drift', t_monotonic: t, drift_s: driftS }
}

test('renderer long-frame spikes only -> renderer-bound', () => {
  const result = classifyCapture(
    bundle({ renderer: rendererStream([longFrame(100, 180), longFrame(400, 260), longFrame(900, 95)]) })
  )

  assert.equal(result.primary, 'renderer-bound')
  assert.deepEqual(result.labels, ['renderer-bound'])
  assert.equal(result.counts.rendererLongFrames, 3)
  assert.equal(result.counts.rendererMaxFrameMs, 260)
})

test('gateway loop stalls only -> gateway-bound', () => {
  const result = classifyCapture(
    bundle({
      gateway: {
        captureId: 'cap-1',
        monotonicAnchorMs: 0,
        dropped: 0,
        events: [loopDrift(10, 0.6), loopDrift(20, 1.4), loopDrift(30, 0.3)]
      }
    })
  )

  assert.equal(result.primary, 'gateway-bound')
  assert.deepEqual(result.labels, ['gateway-bound'])
  assert.equal(result.counts.gatewayLoopDrifts, 3)
  assert.equal(result.counts.gatewayMaxDriftS, 1.4)
})

test('a single severe frame is enough, a single mild one is not', () => {
  assert.deepEqual(classifyCapture(bundle({ renderer: rendererStream([longFrame(1, 420)]) })).labels, ['renderer-bound'])
  assert.deepEqual(classifyCapture(bundle({ renderer: rendererStream([longFrame(1, 60)]) })).labels, ['unclassified'])
})

test('one transport timeout -> ipc-transport-bound', () => {
  const result = classifyCapture(
    bundle({
      main: [{ type: 'transport_error', t: 5, channel: 'hermes:api', route: '/api/sessions', durationMs: 60_000, errorClass: 'timeout' }]
    })
  )

  assert.equal(result.primary, 'ipc-transport-bound')
  assert.equal(result.signals[0].evidence.timeouts, 1)
})

test('a single non-timeout transport error is below the bar', () => {
  const result = classifyCapture(
    bundle({ main: [{ type: 'transport_error', t: 5, channel: 'hermes:api', route: '/api', durationMs: 12, errorClass: 'http_500' }] })
  )

  assert.deepEqual(result.labels, ['unclassified'])
  assert.equal(result.counts.mainTransportErrors, 1)
})

test('heap growth across the capture -> memory-gc-bound', () => {
  const samples = [120, 260, 340, 410].map((usedMb, index) => ({
    type: 'memory_sample',
    t: index * 5_000,
    usedMb,
    totalMb: usedMb + 40,
    limitMb: 4_096
  }))

  const result = classifyCapture(bundle({ renderer: rendererStream(samples) }))

  assert.equal(result.primary, 'memory-gc-bound')
  assert.equal(result.signals[0].evidence.growthMb, 290)
})

test('flat heap samples do not label', () => {
  const samples = [300, 302, 299, 301].map((usedMb, index) => ({
    type: 'memory_sample',
    t: index * 5_000,
    usedMb,
    totalMb: usedMb + 40,
    limitMb: 4_096
  }))

  assert.deepEqual(classifyCapture(bundle({ renderer: rendererStream(samples) })).labels, ['unclassified'])
})

test('commit cost that tracks transcript length -> history-bound', () => {
  // Same flush size throughout; only the transcript grows, and only the cost
  // above the median transcript length is expensive.
  const deltas = [
    { historyMessages: 20, commitMs: 6, writeMs: 2 },
    { historyMessages: 40, commitMs: 8, writeMs: 2 },
    { historyMessages: 900, commitMs: 70, writeMs: 4 },
    { historyMessages: 1_100, commitMs: 90, writeMs: 4 }
  ].map((delta, index) => ({
    type: 'stream_delta_applied',
    t: index * 1_000,
    sessions: 1,
    queuedChars: 400,
    rafGapMs: 8,
    ...delta
  }))

  const result = classifyCapture(bundle({ renderer: rendererStream(deltas) }))

  assert.equal(result.primary, 'history-bound')
  assert.equal(result.signals[0].evidence.samples, 4)
})

test('uniformly cheap flushes do not label history-bound', () => {
  const deltas = [20, 40, 900, 1_100].map((historyMessages, index) => ({
    type: 'stream_delta_applied',
    t: index * 1_000,
    sessions: 1,
    queuedChars: 400,
    rafGapMs: 8,
    historyMessages,
    commitMs: 5,
    writeMs: 1
  }))

  assert.deepEqual(classifyCapture(bundle({ renderer: rendererStream(deltas) })).labels, ['unclassified'])
})

test('multiple mechanisms are multi-labelled, strongest first', () => {
  const result = classifyCapture(
    bundle({
      renderer: rendererStream([longFrame(1, 90), longFrame(2, 95), longFrame(3, 110)]),
      gateway: { captureId: 'cap-1', monotonicAnchorMs: 0, dropped: 0, events: [loopDrift(1, 2.5), loopDrift(2, 3)] }
    })
  )

  assert.deepEqual(result.labels, ['gateway-bound', 'renderer-bound'])
  assert.equal(result.primary, 'gateway-bound')
})

test('an empty capture is unclassified, with zeroed counts', () => {
  const result = classifyCapture(bundle())

  assert.deepEqual(result.labels, ['unclassified'])
  assert.equal(result.primary, null)
  assert.deepEqual(result.signals, [])
  assert.equal(result.counts.rendererLongFrames, 0)
  assert.equal(result.counts.gatewayLoopDrifts, 0)
})
