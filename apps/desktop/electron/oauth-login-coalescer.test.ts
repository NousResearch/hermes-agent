import { expect, test } from 'vitest'

import { loginKeyFor, SilentLoginCoalescer, SilentLoginSuppressedError } from './oauth-login-coalescer'

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
  const a = coalescer.run(KEY, login)
  const b = coalescer.run(KEY, login)
  const c = coalescer.runFor(`${GATEWAY}/api/status`, silent, () => login())

  expect(calls).toBe(1)

  first.resolve('ok')

  await expect(Promise.all([a, b, c])).resolves.toEqual(['ok', 'ok', 'ok'])
  expect(calls).toBe(1)
})

test('a silent login forwards the gateway url and options to a single opener', async () => {
  const coalescer = new SilentLoginCoalescer()
  const seen: Array<{ url: string; options: { silent?: boolean; background?: boolean } }> = []

  const open = (url: string, options: { silent?: boolean; background?: boolean }) => {
    seen.push({ url, options })

    return Promise.resolve('ok')
  }

  const options = { silent: true as const, background: true }

  await Promise.all([
    coalescer.runFor(`${GATEWAY}/api/status`, options, open),
    coalescer.runFor(GATEWAY, options, open)
  ])

  expect(seen).toEqual([{ url: `${GATEWAY}/api/status`, options }])
})

test('a failed silent login is not retried until its cooldown expires', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })
  let calls = 0

  const login = () => {
    calls += 1

    return Promise.reject(new Error('Login window closed before authentication completed.'))
  }

  await expect(coalescer.run(KEY, login)).rejects.toThrow(/Login window closed/)
  expect(calls).toBe(1)

  const suppressed = await coalescer.run(KEY, login).then(
    () => null,
    error => error
  )

  expect(suppressed).toBeInstanceOf(SilentLoginSuppressedError)
  expect((suppressed as SilentLoginSuppressedError).retryAfterMs).toBe(5_000)
  expect(calls).toBe(1)

  clock.advance(5_000)
  await expect(coalescer.run(KEY, login)).rejects.toThrow(/Login window closed/)
  expect(calls).toBe(2)
})

test('the backoff doubles per consecutive failure and stops at the ceiling', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 20_000 })
  const delays: number[] = []

  const fail = () => Promise.reject(new Error('nope'))

  for (let attempt = 0; attempt < 5; attempt += 1) {
    await expect(coalescer.run(KEY, fail)).rejects.toThrow(/nope/)
    delays.push(coalescer.cooldownRemainingMs(KEY))
    clock.advance(coalescer.cooldownRemainingMs(KEY))
  }

  expect(delays).toEqual([5_000, 10_000, 20_000, 20_000, 20_000])
})

test('a successful login clears the backoff', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })

  await expect(coalescer.run(KEY, () => Promise.reject(new Error('nope')))).rejects.toThrow(/nope/)
  clock.advance(5_000)
  await expect(coalescer.run(KEY, () => Promise.resolve('ok'))).resolves.toBe('ok')
  expect(coalescer.cooldownRemainingMs(KEY)).toBe(0)

  await expect(coalescer.run(KEY, () => Promise.reject(new Error('nope')))).rejects.toThrow(/nope/)
  expect(coalescer.cooldownRemainingMs(KEY)).toBe(5_000)
})

test('separate gateways are coalesced independently', async () => {
  const coalescer = new SilentLoginCoalescer()
  let calls = 0

  const login = () => {
    calls += 1

    return Promise.resolve('ok')
  }

  await Promise.all([coalescer.run(KEY, login), coalescer.run(loginKeyFor('https://other.example.test'), login)])
  expect(calls).toBe(2)
})

test('an interactive sign-in is never coalesced and still receives the url', async () => {
  const clock = fakeClock()
  const coalescer = new SilentLoginCoalescer({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })
  const seen: string[] = []

  const open = (url: string) => {
    seen.push(url)

    return Promise.resolve('ok')
  }

  const silentAttempt = deferred<string>()
  const silent = coalescer.runFor(GATEWAY, { silent: true }, () => silentAttempt.promise)

  await Promise.all([
    coalescer.runFor(GATEWAY, { silent: false }, open),
    coalescer.runFor(GATEWAY, { silent: false }, open)
  ])
  expect(seen).toEqual([GATEWAY, GATEWAY])

  silentAttempt.reject(new Error('nope'))
  await expect(silent).rejects.toThrow(/nope/)
  await expect(coalescer.runFor(GATEWAY, { silent: false }, open)).resolves.toBe('ok')
  expect(seen).toEqual([GATEWAY, GATEWAY, GATEWAY])
})

test('the coalescing key is the gateway origin', () => {
  expect(loginKeyFor('https://gateway.example.test')).toBe('https://gateway.example.test')
  expect(loginKeyFor('https://gateway.example.test/some/path')).toBe('https://gateway.example.test')
  expect(loginKeyFor('http://127.0.0.1:9119')).toBe('http://127.0.0.1:9119')
  expect(loginKeyFor('https://gateway.example.test')).not.toBe(loginKeyFor('https://other.example.test'))
})
