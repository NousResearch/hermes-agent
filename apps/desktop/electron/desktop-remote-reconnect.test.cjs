const test = require('node:test')
const assert = require('node:assert/strict')

const {
  createRemoteRevalidateState,
  recordRemoteRevalidateFailure,
  resetRemoteRevalidateState,
  shouldRemoteRevalidateKeepState,
  waitForRemoteHermes
} = require('./desktop-remote-reconnect.cjs')

async function withFakeNow(startMs, fn) {
  const realNow = Date.now
  let nowMs = startMs
  Date.now = () => nowMs
  try {
    await fn({
      advance(ms) {
        nowMs += ms
      }
    })
  } finally {
    Date.now = realNow
  }
}

test('waitForRemoteHermes retries transient probe failures before succeeding', async () => {
  await withFakeNow(10_000, async clock => {
    const sleeps = []
    let attempts = 0

    await waitForRemoteHermes('https://gw.example.com', 'tok', {
      fetcher: async () => {
        attempts += 1
        if (attempts < 3) {
          throw new Error(`miss ${attempts}`)
        }
      },
      maxAttempts: 5,
      timeoutMs: 90_000,
      sleep: async ms => {
        sleeps.push(ms)
        clock.advance(ms)
      }
    })

    assert.equal(attempts, 3)
    assert.deepEqual(sleeps, [500, 1_000])
  })
})

test('waitForRemoteHermes fails after the bounded retry budget', async () => {
  await withFakeNow(20_000, async clock => {
    const sleeps = []

    await assert.rejects(
      () =>
        waitForRemoteHermes('https://gw.example.com', 'tok', {
          fetcher: async () => {
            throw new Error('still restarting')
          },
          maxAttempts: 3,
          timeoutMs: 90_000,
          sleep: async ms => {
            sleeps.push(ms)
            clock.advance(ms)
          }
        }),
      /after 3 attempts: still restarting/
    )

    assert.deepEqual(sleeps, [500, 1_000])
  })
})

test('recordRemoteRevalidateFailure only resets after repeated misses in one window', () => {
  const state = createRemoteRevalidateState()

  assert.equal(shouldRemoteRevalidateKeepState(state), false)
  assert.deepEqual(recordRemoteRevalidateFailure(state, { now: 1_000 }), {
    failures: 1,
    shouldReset: false
  })
  assert.equal(shouldRemoteRevalidateKeepState(state), true)
  assert.deepEqual(recordRemoteRevalidateFailure(state, { now: 5_000 }), {
    failures: 2,
    shouldReset: false
  })
  assert.deepEqual(recordRemoteRevalidateFailure(state, { now: 9_000 }), {
    failures: 3,
    shouldReset: true
  })

  resetRemoteRevalidateState(state)
  assert.equal(shouldRemoteRevalidateKeepState(state), false)
})

test('recordRemoteRevalidateFailure starts a fresh window after the retry budget expires', () => {
  const state = createRemoteRevalidateState()

  assert.deepEqual(recordRemoteRevalidateFailure(state, { now: 1_000 }), {
    failures: 1,
    shouldReset: false
  })
  assert.deepEqual(recordRemoteRevalidateFailure(state, { now: 30_000 }), {
    failures: 1,
    shouldReset: false
  })
})
