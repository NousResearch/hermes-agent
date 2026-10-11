import assert from 'node:assert/strict'

import { test } from 'vitest'

import { loginKeyFor, SilentLoginCoalescer, SilentLoginSuppressedError } from './oauth-login-coalescer'

/** A clock the test drives, so the backoff is provable without sleeping. */
function fakeClock(start = 1_000_000) {
  let now = start

  return {
    now: () => now,
    advance: (ms: number) => {
      now += ms
    }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

const GATEWAY = 'https://gateway.example.test'
const KEY = loginKeyFor(GATEWAY)

test('concurrent silent logins for one gateway share a single attempt', async () => {
  const coalescer = new SilentLoginCoalescer()
  const first = deferred<string>()
  let calls = 0

  const login = () => {
    calls += 1

    return first.promise
  }

  const silent = { silent: true }
  const a = coalescer.runFor(GATEWAY, silent, login)
  const b = coalescer.runFor(GATEWAY, silent, login)
  const c = coalescer.runFor(`${GATEWAY}/some/path`, silent, login)

  assert.equal(calls, 1, 'a burst of refused requests must not open a window each')

  first.resolve('ok')

  assert.deepEqual(await Promise.all([a, b, c]), ['ok', 'ok', 'ok'])
  assert.equal(calls, 1)
})

test('a failed silent login is not retried until its cooldown expires', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })
  let calls = 0

  const login = () => {
    calls += 1

    return Promise.reject(new Error('Login window closed before authentication completed.'))
  }

  await assert.rejects(() => coalescer.run(KEY, login), /Login window closed/)
  assert.equal(calls, 1)

  const suppressed = await coalescer.run(KEY, login).then(
    () => null,
    error => error
  )

  assert.ok(suppressed instanceof SilentLoginSuppressedError)
  assert.equal(suppressed.retryAfterMs, 5_000)
  assert.equal(calls, 1, 'the cooldown must hold the next attempt, not open another window')

  clock.advance(5_000)
  await assert.rejects(() => coalescer.run(KEY, login), /Login window closed/)
  assert.equal(calls, 2, 'once the cooldown expires the attempt is allowed again')
})

test('the backoff doubles per consecutive failure and stops at the ceiling', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 20_000 })
  const delays: number[] = []

  const fail = () => Promise.reject(new Error('nope'))

  for (let attempt = 0; attempt < 5; attempt += 1) {
    await assert.rejects(() => coalescer.run(KEY, fail))
    delays.push(coalescer.cooldownRemainingMs(KEY))
    clock.advance(coalescer.cooldownRemainingMs(KEY))
  }

  assert.deepEqual(delays, [5_000, 10_000, 20_000, 20_000, 20_000])
})

test('a successful login clears the backoff for the next genuine expiry', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })

  await assert.rejects(() => coalescer.run(KEY, () => Promise.reject(new Error('nope'))))
  clock.advance(5_000)
  await coalescer.run(KEY, () => Promise.resolve('ok'))
  assert.equal(coalescer.cooldownRemainingMs(KEY), 0)

  await assert.rejects(() => coalescer.run(KEY, () => Promise.reject(new Error('nope'))))
  assert.equal(coalescer.cooldownRemainingMs(KEY), 5_000, 'the failure count restarts at the base delay')
})

test('separate gateways are coalesced independently', async () => {
  const coalescer = new SilentLoginCoalescer()
  const other = 'https://other.example.test'
  let calls = 0

  const login = () => {
    calls += 1

    return Promise.resolve('ok')
  }

  await Promise.all([coalescer.run(KEY, login), coalescer.run(loginKeyFor(other), login)])
  assert.equal(calls, 2, 'one gateway cooling down must not suppress another')
})

test('an interactive sign-in is never coalesced or held back', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })
  const interactive = { silent: false }
  let calls = 0

  const login = () => {
    calls += 1

    return Promise.resolve('ok')
  }

  // The user's own click opens a window even while a silent attempt is in flight…
  const silentAttempt = deferred<string>()
  const silent = coalescer.runFor(GATEWAY, { silent: true }, () => silentAttempt.promise)
  await Promise.all([coalescer.runFor(GATEWAY, interactive, login), coalescer.runFor(GATEWAY, interactive, login)])
  assert.equal(calls, 2, 'each click gets its own window')

  // …and a cooldown left by that failed silent attempt does not block it either.
  silentAttempt.reject(new Error('nope'))
  await assert.rejects(() => silent)
  await coalescer.runFor(GATEWAY, interactive, login)
  assert.equal(calls, 3)
})

test('the coalescing key is the gateway origin', () => {
  assert.equal(loginKeyFor('https://gateway.example.test'), 'https://gateway.example.test')
  assert.equal(loginKeyFor('https://gateway.example.test/some/path'), 'https://gateway.example.test')
  assert.equal(loginKeyFor('http://127.0.0.1:9119'), 'http://127.0.0.1:9119')
  assert.notEqual(loginKeyFor('https://gateway.example.test'), loginKeyFor('https://other.example.test'))
})
