import assert from 'node:assert/strict'

import { test } from 'vitest'

import { CDP } from './cdp.mjs'
import { aggregateTraceEvents, layerSnapshot, watchLayerTreeRate, withCompositingTrace } from './compositing.mjs'

const complete = (name, durUs) => ({ name, ph: 'X', dur: durUs, ts: 0 })

test('attributes a phase as a share of the trace window', () => {
  // Synthetic durations exercise proportional aggregation, not measured performance.
  const phases = aggregateTraceEvents([complete('Layerize', 6_871_000)], 15_000)

  assert.equal(phases.Layerize.pct, 45.8)
  assert.equal(phases.Layerize.ms, 6871)
  assert.equal(phases.Layerize.n, 1)
})

test('sums repeated events and keeps the worst single pass', () => {
  const phases = aggregateTraceEvents(
    [complete('Layerize', 4000), complete('Layerize', 10_000), complete('Layerize', 6000)],
    1000
  )

  assert.equal(phases.Layerize.n, 3)
  assert.equal(phases.Layerize.ms, 20)
  assert.equal(phases.Layerize.max_ms, 10)
})

test('reports every tracked phase, including ones the trace never emitted', () => {
  // A scenario that never painted must still publish paint_pct: 0, otherwise
  // the baseline gate sees a missing key instead of a real zero.
  const phases = aggregateTraceEvents([complete('Layerize', 1000)], 1000)

  assert.equal(phases.Paint.pct, 0)
  assert.equal(phases.Paint.n, 0)
  assert.equal(phases.Commit.ms, 0)
})

test('ignores instant and async events that carry no duration', () => {
  // Begin/End pairs and instant marks share the stream; counting them as
  // complete events would inflate every phase.
  const phases = aggregateTraceEvents(
    [
      { name: 'Layerize', ph: 'B', ts: 0 },
      { name: 'Layerize', ph: 'E', ts: 5000 },
      { name: 'Layerize', ph: 'I', ts: 1 },
      complete('Layerize', 2000)
    ],
    1000
  )

  assert.equal(phases.Layerize.n, 1)
  assert.equal(phases.Layerize.ms, 2)
})

test('survives a zero-length window without dividing by zero', () => {
  const phases = aggregateTraceEvents([complete('Layerize', 1000)], 0)

  assert.ok(Number.isFinite(phases.Layerize.pct))
})

test('ignores untracked phases so the metric set stays stable', () => {
  const phases = aggregateTraceEvents([complete('SomeFutureBlinkPhase', 9_000_000)], 1000)

  assert.equal(phases.Layerize.ms, 0)
  assert.equal(Object.hasOwn(phases, 'SomeFutureBlinkPhase'), false)
})

function fakeCdp() {
  const listeners = new Map()
  const removed = []

  return {
    listeners,
    removed,
    on(method, handler) {
      const handlers = listeners.get(method) ?? []
      handlers.push(handler)
      listeners.set(method, handlers)

      return () => {
        removed.push(method)
        listeners.set(method, (listeners.get(method) ?? []).filter(candidate => candidate !== handler))
      }
    },
    async send(method) {
      if (method === 'Tracing.end') {
        for (const handler of listeners.get('Tracing.tracingComplete') ?? []) {
          handler()
        }
      }

      return {}
    }
  }
}

test('cleans compositing CDP listeners after each scoped measurement', async () => {
  const cdp = fakeCdp()

  for (let i = 0; i < 2; i++) {
    await withCompositingTrace(cdp, async () => undefined)
    await layerSnapshot(cdp, { settleMs: 0 })
    const stop = watchLayerTreeRate(cdp)
    stop()
  }

  assert.deepEqual(cdp.removed.sort(), [
    'LayerTree.layerTreeDidChange',
    'LayerTree.layerTreeDidChange',
    'LayerTree.layerTreeDidChange',
    'LayerTree.layerTreeDidChange',
    'Tracing.dataCollected',
    'Tracing.dataCollected',
    'Tracing.tracingComplete',
    'Tracing.tracingComplete'
  ])
  assert.equal(cdp.listeners.get('Tracing.dataCollected').length, 0)
  assert.equal(cdp.listeners.get('Tracing.tracingComplete').length, 0)
  assert.equal(cdp.listeners.get('LayerTree.layerTreeDidChange').length, 0)
})

test('cleans trace listeners when the measured scenario rejects', async () => {
  const cdp = fakeCdp()

  await assert.rejects(
    withCompositingTrace(cdp, async () => {
      throw new Error('scenario failed')
    }),
    /scenario failed/
  )

  assert.equal(cdp.listeners.get('Tracing.dataCollected').length, 0)
  assert.equal(cdp.listeners.get('Tracing.tracingComplete').length, 0)
})

test('CDP subscriptions return idempotent unsubscribe functions', () => {
  const cdp = { listeners: new Map() }
  const handler = () => undefined
  const unsubscribe = CDP.prototype.on.call(cdp, 'Tracing.dataCollected', handler)

  assert.equal(cdp.listeners.get('Tracing.dataCollected').length, 1)
  unsubscribe()
  unsubscribe()

  assert.equal(cdp.listeners.has('Tracing.dataCollected'), false)
})

test('repeated unsubscribe preserves a duplicate callback registration', () => {
  const cdp = new CDP(null)
  const handler = () => undefined
  const first = cdp.on('Tracing.dataCollected', handler)
  const second = cdp.on('Tracing.dataCollected', handler)

  first()
  first()
  assert.deepEqual(cdp.listeners.get('Tracing.dataCollected'), [handler])
  second()
  second()
  assert.equal(cdp.listeners.has('Tracing.dataCollected'), false)
})
