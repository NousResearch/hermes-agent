import assert from 'node:assert/strict'
import test from 'node:test'

import {
  BUDGET_EXCEEDED,
  createRemoteAvailability,
  isNothingListeningError,
  raceWithBudget,
  spliceRemoteSessions,
  waitForBackendReady
} from './remote-sessions'

const tick = () => new Promise(resolve => setImmediate(resolve))

function codedError(code, message = code) {
  const error: any = new Error(message)
  error.code = code
  return error
}

// ---------------------------------------------------------------------------
// waitForBackendReady
// ---------------------------------------------------------------------------

test('waitForBackendReady resolves as soon as the probe succeeds', async () => {
  let calls = 0
  await waitForBackendReady(async () => {
    calls += 1
  }, { deadlineMs: 1_000, intervalMs: 0 })
  assert.equal(calls, 1)
})

test('waitForBackendReady retries refused connections for local boots (no failFast)', async () => {
  let calls = 0
  await waitForBackendReady(
    async () => {
      calls += 1
      if (calls < 3) throw codedError('ECONNREFUSED', 'connect ECONNREFUSED 127.0.0.1:9120')
    },
    { deadlineMs: 5_000, intervalMs: 0 }
  )
  assert.equal(calls, 3)
})

test('waitForBackendReady with failFast rejects on the first refused connection', async () => {
  let calls = 0
  await assert.rejects(
    waitForBackendReady(
      async () => {
        calls += 1
        throw codedError('ECONNREFUSED', 'connect ECONNREFUSED 127.0.0.1:19119')
      },
      { deadlineMs: 45_000, intervalMs: 0, failFast: true }
    ),
    /not listening/
  )
  assert.equal(calls, 1)
})

test('waitForBackendReady with failFast still retries non-final errors until the deadline', async () => {
  // A timeout/5xx means the server may still be coming up — failFast must not
  // short-circuit those, only "nothing is listening" errors.
  let now = 0
  let calls = 0
  await assert.rejects(
    waitForBackendReady(
      async () => {
        calls += 1
        throw new Error('Timed out connecting to Hermes backend after 1500ms')
      },
      {
        deadlineMs: 3_000,
        intervalMs: 500,
        failFast: true,
        now: () => now,
        sleep: async ms => {
          now += ms
        }
      }
    ),
    /did not become ready/
  )
  assert.ok(calls >= 2, `expected multiple attempts, got ${calls}`)
})

test('waitForBackendReady enforces the deadline with an injected clock', async () => {
  let now = 0
  let calls = 0
  await assert.rejects(
    waitForBackendReady(
      async () => {
        calls += 1
        throw new Error('boom')
      },
      {
        deadlineMs: 2_000,
        intervalMs: 500,
        now: () => now,
        sleep: async ms => {
          now += ms
        }
      }
    ),
    /did not become ready: boom/
  )
  assert.equal(calls, 4) // t=0, 500, 1000, 1500 — t=2000 is past the deadline
})

// ---------------------------------------------------------------------------
// createRemoteAvailability
// ---------------------------------------------------------------------------

test('availability cooldown opens after markDown and expires after cooldownMs', () => {
  let now = 0
  const availability = createRemoteAvailability({ cooldownMs: 30_000, now: () => now })

  assert.equal(availability.inCooldown('taro'), false)
  availability.markDown('taro', 'connect ECONNREFUSED')
  assert.equal(availability.inCooldown('taro'), true)

  now = 29_999
  assert.equal(availability.inCooldown('taro'), true)

  now = 30_000
  assert.equal(availability.inCooldown('taro'), false)
})

test('availability markUp and clear lift the cooldown immediately', () => {
  let now = 0
  const availability = createRemoteAvailability({ cooldownMs: 30_000, now: () => now })

  availability.markDown('taro')
  availability.markUp('taro')
  assert.equal(availability.inCooldown('taro'), false)

  availability.markDown('taro')
  availability.clear()
  assert.equal(availability.inCooldown('taro'), false)
})

test('availability keys are trimmed and empty keys ignored', () => {
  const availability = createRemoteAvailability({ cooldownMs: 30_000, now: () => 0 })
  availability.markDown(' taro ')
  assert.equal(availability.inCooldown('taro'), true)
  availability.markDown('')
  assert.equal(availability.inCooldown(''), false)
})

// ---------------------------------------------------------------------------
// raceWithBudget
// ---------------------------------------------------------------------------

test('raceWithBudget returns the settled value within budget and clears its timer', async () => {
  let cleared = false
  const result = await raceWithBudget(
    Promise.resolve('rows'),
    1_000,
    setTimeout,
    timer => {
      cleared = true
      clearTimeout(timer)
    }
  )
  assert.equal(result, 'rows')
  assert.equal(cleared, true)
})

test('raceWithBudget resolves BUDGET_EXCEEDED when the promise is slower than the budget', async () => {
  const never = new Promise(() => {})
  const result = await raceWithBudget(never, 5)
  assert.equal(result, BUDGET_EXCEEDED)
})

// ---------------------------------------------------------------------------
// isNothingListeningError
// ---------------------------------------------------------------------------

test('isNothingListeningError recognizes dead-listener codes only', () => {
  assert.equal(isNothingListeningError(codedError('ECONNREFUSED')), true)
  assert.equal(isNothingListeningError(codedError('ENOTFOUND')), true)
  assert.equal(isNothingListeningError(codedError('ETIMEDOUT')), false)
  assert.equal(isNothingListeningError(new Error('plain')), false)
  assert.equal(isNothingListeningError(null), false)
})

// ---------------------------------------------------------------------------
// spliceRemoteSessions
// ---------------------------------------------------------------------------

const row = (id, profile, lastActive) => ({ id, profile, last_active: lastActive, started_at: lastActive })

function baseFixture() {
  return {
    sessions: [row('local-1', 'default', 100), row('stale-taro', 'taro', 50)],
    total: 12, // 10 local + 2 stale local rows for taro
    profile_totals: { default: 10, taro: 2 }
  }
}

test('splice swaps a healthy remote profile rows in, re-sorts, and fixes totals', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  const result = await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    fetchRemote: async () => ({ sessions: [row('taro-1', 'taro', 200), row('taro-2', 'taro', 10)], total: 7 })
  })

  assert.deepEqual(result.sessions.map(s => s.id), ['taro-1', 'local-1', 'taro-2'])
  assert.equal(result.total, 17) // 10 local + 7 real remote
  assert.equal(result.profile_totals.taro, 7)
  // stale local copy of the remote's rows must not survive the splice
  assert.ok(!result.sessions.some(s => s.id === 'stale-taro'))
})

test('splice skips a remote in cooldown instantly without calling fetchRemote', async () => {
  let now = 0
  const availability = createRemoteAvailability({ cooldownMs: 30_000, now: () => now })
  availability.markDown('taro', 'connect ECONNREFUSED')

  let fetchCalls = 0
  const result = await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    fetchRemote: async () => {
      fetchCalls += 1
      return { sessions: [], total: 0 }
    }
  })

  assert.equal(fetchCalls, 0)
  assert.deepEqual(result.sessions.map(s => s.id), ['local-1'])
  assert.equal(result.total, 10) // remote contributes nothing
  assert.equal('taro' in result.profile_totals, false)
})

test('splice tolerates a rejecting remote: local rows render, totals drop the remote', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  const result = await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    fetchRemote: async () => {
      throw codedError('ECONNREFUSED', 'connect ECONNREFUSED 127.0.0.1:19119')
    }
  })

  assert.deepEqual(result.sessions.map(s => s.id), ['local-1'])
  assert.equal(result.total, 10)
  assert.equal('taro' in result.profile_totals, false)
})

test('splice gives up after the budget but leaves the slow fetch running for the next refresh', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  let settleLate
  const slow = new Promise(resolve => {
    settleLate = resolve
  })

  const result = await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    coldBudgetMs: 5,
    fetchRemote: () => slow
  })

  assert.deepEqual(result.sessions.map(s => s.id), ['local-1'])
  assert.equal('taro' in result.profile_totals, false)

  // Late settle must not throw or corrupt anything (fetch kept running).
  settleLate({ sessions: [row('taro-1', 'taro', 200)], total: 1 })
  await tick()
})

test('splice late REJECTION after budget does not produce an unhandled rejection', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  let rejectLate
  const slow = new Promise((_resolve, reject) => {
    rejectLate = reject
  })

  const unhandled = []
  const onUnhandled = reason => unhandled.push(reason)
  process.on('unhandledRejection', onUnhandled)

  try {
    await spliceRemoteSessions({
      base: baseFixture(),
      remoteProfiles: ['taro'],
      limit: 10,
      offset: 0,
      order: 'last_active',
      availability,
      coldBudgetMs: 5,
      fetchRemote: () => slow
    })

    rejectLate(codedError('ECONNREFUSED', 'late failure'))
    await tick()
    await tick()
    assert.deepEqual(unhandled, [])
  } finally {
    process.off('unhandledRejection', onUnhandled)
  }
})

test('splice clears the cooldown via markUp when a remote answers again', async () => {
  let now = 0
  const availability = createRemoteAvailability({ cooldownMs: 30_000, now: () => now })
  availability.markDown('taro')
  now = 31_000 // cooldown expired → splice probes again

  const result = await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    fetchRemote: async () => ({ sessions: [row('taro-1', 'taro', 200)], total: 1 })
  })

  assert.equal(result.profile_totals.taro, 1)
  assert.equal(availability.inCooldown('taro'), false)
})

test('splice uses the warm budget for warm profiles', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  const budgets = []

  await spliceRemoteSessions({
    base: baseFixture(),
    remoteProfiles: ['taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    isWarm: () => true,
    coldBudgetMs: 1,
    warmBudgetMs: 50,
    setTimeoutFn: (fn, ms) => {
      budgets.push(ms)
      return setTimeout(fn, ms)
    },
    fetchRemote: async () => ({ sessions: [], total: 0 })
  })

  assert.deepEqual(budgets, [50])
})

test('splice windows the merged list to limit/offset', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  const base = {
    sessions: [row('l1', 'default', 400), row('l2', 'default', 300)],
    total: 2,
    profile_totals: { default: 2 }
  }

  const result = await spliceRemoteSessions({
    base,
    remoteProfiles: ['taro'],
    limit: 2,
    offset: 1,
    order: 'last_active',
    availability,
    fetchRemote: async () => ({ sessions: [row('t1', 'taro', 500), row('t2', 'taro', 100)], total: 2 })
  })

  // recency order: t1(500), l1(400), l2(300), t2(100) → offset 1, limit 2
  assert.deepEqual(result.sessions.map(s => s.id), ['l1', 'l2'])
  assert.equal(result.total, 4)
})

test('splice handles several remotes independently (one down, one up)', async () => {
  const availability = createRemoteAvailability({ now: () => 0 })
  availability.markDown('dead-rig')

  const result = await spliceRemoteSessions({
    base: {
      sessions: [row('l1', 'default', 100)],
      total: 1,
      profile_totals: { default: 1 }
    },
    remoteProfiles: ['dead-rig', 'taro'],
    limit: 10,
    offset: 0,
    order: 'last_active',
    availability,
    fetchRemote: async name => {
      assert.equal(name, 'taro')
      return { sessions: [row('t1', 'taro', 200)], total: 3 }
    }
  })

  assert.deepEqual(result.sessions.map(s => s.id), ['t1', 'l1'])
  assert.equal(result.total, 4)
  assert.equal(result.profile_totals.taro, 3)
  assert.equal('dead-rig' in result.profile_totals, false)
})
