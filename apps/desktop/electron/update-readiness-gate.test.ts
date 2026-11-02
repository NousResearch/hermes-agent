import assert from 'node:assert/strict'

import { test } from 'vitest'

import { type ActiveWork, NO_ACTIVE_WORK } from './quit-guard'
import { mergeUpdateReadiness, UpdateWhenIdle } from './update-readiness'
import {
  applyUpdatesWhenIdle,
  busyApplyResult,
  cancelUpdateWhenIdle,
  createUpdateReadinessGate,
  type UpdateWaitEnding
} from './update-readiness-gate'

interface FakeBackend {
  baseUrl: string
  idle: unknown
  fence: 'grant' | 'refuse' | 'missing'
  calls: string[]
}

function backend(idle: unknown, fence: FakeBackend['fence'] = 'grant', baseUrl = 'http://127.0.0.1:1'): FakeBackend {
  return { baseUrl, idle, fence, calls: [] }
}

function gateFor(
  backends: FakeBackend[],
  work: () => ActiveWork = () => NO_ACTIVE_WORK,
  every?: (ms: number, tick: () => void) => () => void
) {
  return createUpdateReadinessGate<FakeBackend>({
    activeWork: work,
    every,
    localBackends: async () => backends,
    log: () => undefined,
    request: async (b, path, opts) => {
      const body = opts.body as { action?: string; token?: string } | undefined
      b.calls.push(body?.action ? `${path}:${body.action}${body.token ? `:${body.token}` : ''}` : path)

      if (path === '/api/health/idle') {
        if (b.idle instanceof Error) {
          throw b.idle
        }

        return b.idle
      }

      if (b.fence === 'missing') {
        throw Object.assign(new Error('404: Not Found'), { statusCode: 404 })
      }

      if (body?.action === 'prepare') {
        return b.fence === 'grant' ? { ok: true, idle: true, token: `tok-${b.baseUrl}` } : { ok: false, idle: false }
      }

      return { ok: true }
    }
  })
}

const SESSION_TURN = { ok: true, idle: false, reason: 'turn_in_flight', detail: 'session:abc' }
const IDLE = { ok: true, idle: true, reason: null }

test('a running chat refuses the hand-off, names it, and takes no admission fence', async () => {
  const b = backend(SESSION_TURN)
  const freeze = await gateFor([b], () => ({ count: 1, titles: ['Refactor auth'] })).freeze()

  assert.equal(freeze.ok, false)
  assert.deepEqual(freeze.reasons, [{ kind: 'chats', count: 1, titles: ['Refactor auth'] }])
  assert.deepEqual(b.calls, ['/api/health/idle'])
})

test('a backend that cannot prove idle counts as busy (never assumed idle)', async () => {
  for (const idle of [{ ok: true, idle: null, reason: 'turn_probe_unavailable' }, new Error('ECONNRESET')]) {
    const freeze = await gateFor([backend(idle)]).freeze()

    assert.deepEqual(freeze.reasons, [{ kind: 'unknown' }])
  }
})

test('cron runs and delegations are named from the backend ledger', () => {
  const readiness = mergeUpdateReadiness(NO_ACTIVE_WORK, [
    { idle: false, reason: 'turn_in_flight', detail: 'cron:nightly-digest' },
    { idle: false, reason: 'turn_in_flight', detail: 'delegation' },
    { idle: false, reason: 'awaiting_human_input' }
  ])

  assert.equal(readiness.ready, false)
  assert.deepEqual(readiness.reasons, [
    { kind: 'cron', jobs: ['nightly-digest'] },
    { kind: 'delegation' },
    { kind: 'input' }
  ])
})

test('an idle machine freezes every backend, and release reopens exactly those fences', async () => {
  const a = backend(IDLE, 'grant', 'http://a')
  const old = backend(IDLE, 'missing', 'http://old')
  const freeze = await gateFor([a, old]).freeze()

  assert.equal(freeze.ok, true)
  assert.ok(a.calls.includes('/api/health/retirement:prepare'))

  await freeze.release()

  assert.ok(a.calls.includes('/api/health/retirement:cancel:tok-http://a'))
  assert.ok(!old.calls.some(call => call.includes('cancel')))
})

test('work that starts between the idle read and the fence releases fences already taken', async () => {
  const first = backend(IDLE, 'grant', 'http://first')
  const racing = backend(IDLE, 'refuse', 'http://racing')
  const freeze = await gateFor([first, racing]).freeze()

  assert.equal(freeze.ok, false)
  assert.ok(first.calls.includes('/api/health/retirement:cancel:tok-http://first'))
})

function clock() {
  let now = 0

  return { now: () => now, sleep: async (ms: number) => void (now += ms) }
}

test('update-when-idle fires only after the turn ends and readiness holds twice', async () => {
  const reads = [false, false, true, false, true, true] // busy, busy, a 1-poll gap, busy, idle, idle
  const fired: number[] = []
  const published: unknown[] = []
  const time = clock()

  const result = await applyUpdatesWhenIdle({
    scheduler: new UpdateWhenIdle(),
    read: async () =>
      reads.shift()
        ? { ready: true, reasons: [] }
        : { ready: false, reasons: [{ kind: 'chats', count: 1, titles: [] }] },
    applyGated: async () => {
      fired.push(time.now())

      return { ok: true, handedOff: true }
    },
    emitWaiting: reasons => published.push(reasons),
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 60_000,
    ...time
  })

  assert.deepEqual(result, { ok: true, handedOff: true })
  assert.equal(fired.length, 1)
  assert.equal(reads.length, 0)
  assert.ok(published.length > 0)
})

test('a busy answer from the final gate keeps waiting instead of handing off', async () => {
  const gate = [busyApplyResult([{ kind: 'cron', jobs: ['j1'] }]), { ok: true, handedOff: true }]
  const time = clock()

  const result = await applyUpdatesWhenIdle({
    scheduler: new UpdateWhenIdle(),
    read: async () => ({ ready: true, reasons: [] }),
    applyGated: async () => gate.shift()!,
    emitWaiting: () => undefined,
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 60_000,
    ...time
  })

  assert.deepEqual(result, { ok: true, handedOff: true })
  assert.equal(gate.length, 0)
})

test('cancel stops the wait without ever handing off', async () => {
  const scheduler = new UpdateWhenIdle<{ ok: boolean }>()
  let fired = false

  const pending = applyUpdatesWhenIdle({
    scheduler,
    read: async () => ({ ready: false, reasons: [{ kind: 'delegation' }] }),
    applyGated: async () => {
      fired = true

      return { ok: true }
    },
    emitWaiting: () => undefined,
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 5,
    maxWaitMs: 60_000
  })

  await new Promise(resolve => setTimeout(resolve, 20))
  assert.equal(scheduler.cancel(), true)
  assert.deepEqual(await pending, { ok: false, error: 'cancelled' })
  assert.equal(fired, false)
})

test('the wait is bounded: still busy at the deadline gives up and names what held it', async () => {
  const time = clock()
  let fired = false

  const result = await applyUpdatesWhenIdle({
    scheduler: new UpdateWhenIdle(),
    read: async () => ({ ready: false, reasons: [{ kind: 'cron', jobs: ['backup'] }] }),
    applyGated: async () => {
      fired = true

      return { ok: true }
    },
    emitWaiting: () => undefined,
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 10_000,
    ...time
  })

  assert.equal(fired, false)
  assert.equal((result as { error?: string }).error, 'idle-timeout')
  assert.deepEqual((result as { waitingFor?: unknown }).waitingFor, [{ kind: 'cron', jobs: ['backup'] }])
})

test('a wait whose idle read lands past the deadline never hands off', async () => {
  let now = 0
  let reads = 0
  let fired = false

  const result = await applyUpdatesWhenIdle({
    scheduler: new UpdateWhenIdle(),
    read: async () => {
      reads += 1
      // One idle read, then the laptop sleeps through the bound: the second idle read comes too late.
      now = reads === 2 ? 2 * 60 * 60 * 1000 : now

      return { ready: true, reasons: [] }
    },
    applyGated: async () => {
      fired = true

      return { ok: true }
    },
    emitWaiting: () => undefined,
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 60_000,
    now: () => now,
    sleep: async () => undefined
  })

  assert.equal(fired, false)
  assert.equal((result as { error?: string }).error, 'idle-timeout')
})

test('cancel during the final gate reaches the hand-off before it tears anything down', async () => {
  const scheduler = new UpdateWhenIdle<{ ok: boolean; error?: string }>()
  let tornDown = false

  const result = await applyUpdatesWhenIdle({
    scheduler,
    read: async () => ({ ready: true, reasons: [] }),
    applyGated: async stillWanted => {
      scheduler.cancel() // the user clicks Cancel while the gate awaits its backends

      if (!stillWanted()) {
        return { ok: false, error: 'cancelled' }
      }

      tornDown = true

      return { ok: true }
    },
    emitWaiting: () => undefined,
    emitEnded: () => undefined,
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 60_000,
    ...clock()
  })

  assert.equal(tornDown, false)
  assert.deepEqual(result, { ok: false, error: 'cancelled' })
})

test('a held fence is renewed while the hand-off prepares, and release stops renewing before it reopens', async () => {
  const b = backend(IDLE, 'grant', 'http://a')
  const ticks: Array<() => void> = []
  let stopped = false

  const freeze = await gateFor([b], undefined, (_ms, tick) => {
    ticks.push(tick)

    return () => {
      stopped = true
    }
  }).freeze()

  assert.equal(freeze.ok, true)
  assert.equal(ticks.length, 1)

  // A long download: three heartbeats, each renewing the permit it holds.
  for (let i = 0; i < 3; i++) {
    ticks[0]()
  }

  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(b.calls.filter(call => call === '/api/health/retirement:renew:tok-http://a').length, 3)

  await freeze.release() // the download failed or the user cancelled

  assert.equal(stopped, true)
  assert.ok(b.calls.includes('/api/health/retirement:cancel:tok-http://a'))
})

test('every window hears a wait end: cancel broadcasts at once, the deadline broadcasts its reasons', async () => {
  const endings: UpdateWaitEnding[] = []
  const scheduler = new UpdateWhenIdle<{ ok: boolean }>()

  assert.deepEqual(
    cancelUpdateWhenIdle(scheduler, ending => endings.push(ending)),
    { cancelled: false }
  )
  assert.deepEqual(endings, [])

  const pending = applyUpdatesWhenIdle({
    scheduler,
    read: async () => ({ ready: false, reasons: [{ kind: 'delegation' }] }),
    applyGated: async () => ({ ok: true }),
    emitWaiting: () => undefined,
    emitEnded: ending => endings.push(ending),
    log: () => undefined,
    pollMs: 5,
    maxWaitMs: 60_000
  })

  await new Promise(resolve => setTimeout(resolve, 20))
  assert.deepEqual(
    cancelUpdateWhenIdle(scheduler, ending => endings.push(ending)),
    { cancelled: true }
  )
  assert.deepEqual(endings, [{ stage: 'idle' }])
  await pending

  endings.length = 0
  await applyUpdatesWhenIdle({
    scheduler: new UpdateWhenIdle(),
    read: async () => ({ ready: false, reasons: [{ kind: 'cron', jobs: ['backup'] }] }),
    applyGated: async () => ({ ok: true }),
    emitWaiting: () => undefined,
    emitEnded: ending => endings.push(ending),
    log: () => undefined,
    pollMs: 1000,
    maxWaitMs: 10_000,
    ...clock()
  })

  assert.equal(endings.length, 1)
  assert.equal(endings[0].stage, 'error')
  assert.deepEqual((endings[0] as { waitingFor?: unknown }).waitingFor, [{ kind: 'cron', jobs: ['backup'] }])
})
