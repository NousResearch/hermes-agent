import { readFileSync } from 'node:fs'

import { afterEach, expect, test, vi } from 'vitest'

import {
  type LoginAttempt,
  loginFlightKey,
  LoginJoinTimeoutError,
  OauthLoginFlights,
  SilentLoginSuppressedError
} from './oauth-login-flights'

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

/** A stand-in login window: counts how many were opened and revealed. */
function fakeWindows<T>(outcome: () => Promise<T>) {
  const stats = { opened: 0, revealed: 0 }

  const start = (): LoginAttempt<T> => {
    stats.opened += 1

    return {
      result: outcome(),
      reveal: () => {
        stats.revealed += 1
      }
    }
  }

  return { stats, start }
}

const JAR = 'persist:hermes-remote-oauth'
const GATEWAY = 'https://gateway.example.test'
const KEY = loginFlightKey(JAR, GATEWAY)

afterEach(() => {
  vi.useRealTimers()
})

test('a burst of background 401s opens one window for the gateway', async () => {
  const flights = new OauthLoginFlights()
  const login = deferred<string>()
  const windows = fakeWindows(() => login.promise)

  const burst = Array.from({ length: 10 }, () => flights.run(KEY, 'background', windows.start))

  expect(windows.stats.opened).toBe(1)
  login.resolve('ok')
  await expect(Promise.all(burst)).resolves.toEqual(Array(10).fill('ok'))
  expect(windows.stats).toEqual({ opened: 1, revealed: 0 })
})

test('a failed background login is held until its cooldown expires', async () => {
  const clock = fakeClock()
  const flights = new OauthLoginFlights({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 60_000 })
  const windows = fakeWindows(() => Promise.reject(new Error('Login window closed before authentication completed.')))

  await expect(flights.run(KEY, 'background', windows.start)).rejects.toThrow(/Login window closed/)

  const held = await flights.run(KEY, 'background', windows.start).catch(error => error)

  expect(held).toBeInstanceOf(SilentLoginSuppressedError)
  expect((held as SilentLoginSuppressedError).retryAfterMs).toBe(5_000)
  expect(windows.stats.opened).toBe(1)

  // After a wake the next attempt runs on its own once the hold lapses.
  clock.advance(5_000)
  await expect(flights.run(KEY, 'background', windows.start)).rejects.toThrow(/Login window closed/)
  expect(windows.stats.opened).toBe(2)
})

test('the backoff doubles per consecutive failure, stops at the ceiling, and clears on success', async () => {
  const clock = fakeClock()
  const flights = new OauthLoginFlights({ now: clock.now, baseDelayMs: 5_000, maxDelayMs: 20_000 })
  const fail = fakeWindows(() => Promise.reject(new Error('nope')))
  const delays: number[] = []

  for (let attempt = 0; attempt < 5; attempt += 1) {
    await expect(flights.run(KEY, 'background', fail.start)).rejects.toThrow(/nope/)
    delays.push(flights.cooldownRemainingMs(KEY))
    clock.advance(flights.cooldownRemainingMs(KEY))
  }

  expect(delays).toEqual([5_000, 10_000, 20_000, 20_000, 20_000])

  await expect(flights.run(KEY, 'background', fakeWindows(async () => 'ok').start)).resolves.toBe('ok')
  expect(flights.cooldownRemainingMs(KEY)).toBe(0)
})

test('a click while a hidden attempt runs reveals that window instead of opening a second', async () => {
  const flights = new OauthLoginFlights()
  const login = deferred<string>()
  const windows = fakeWindows(() => login.promise)

  const background = flights.run(KEY, 'background', windows.start)
  const click = flights.run(KEY, 'foreground', windows.start)
  const repeatClick = flights.run(KEY, 'foreground', windows.start)

  expect(windows.stats).toEqual({ opened: 1, revealed: 2 })

  login.resolve('ok')
  await expect(Promise.all([background, click, repeatClick])).resolves.toEqual(['ok', 'ok', 'ok'])
})

test('a click is never held by a background cooldown', async () => {
  const clock = fakeClock()
  const flights = new OauthLoginFlights({ now: clock.now })

  await expect(
    flights.run(KEY, 'background', fakeWindows(() => Promise.reject(new Error('nope'))).start)
  ).rejects.toThrow(/nope/)
  expect(flights.cooldownRemainingMs(KEY)).toBeGreaterThan(0)

  const click = fakeWindows(async () => 'ok')

  await expect(flights.run(KEY, 'foreground', click.start)).resolves.toBe('ok')
  expect(click.stats.opened).toBe(1)
  expect(flights.cooldownRemainingMs(KEY)).toBe(0)
})

test('background callers stop waiting on a window the user left open', async () => {
  vi.useFakeTimers()

  const flights = new OauthLoginFlights({ joinTimeoutMs: 15_000 })
  const login = deferred<string>()
  const windows = fakeWindows(() => login.promise)

  const owner = flights.run(KEY, 'foreground', windows.start)
  const joiner = flights.run(KEY, 'background', windows.start)
  const joinerOutcome = joiner.catch(error => error)

  await vi.advanceTimersByTimeAsync(15_000)
  expect(await joinerOutcome).toBeInstanceOf(LoginJoinTimeoutError)

  // The user's own window is untouched and still completes.
  expect(flights.isRunning(KEY)).toBe(true)
  login.resolve('ok')
  await expect(owner).resolves.toBe('ok')
  expect(windows.stats.opened).toBe(1)
})

test('the same origin on two cookie jars gets two independent flows', async () => {
  const flights = new OauthLoginFlights()
  const primary = deferred<string>()
  const other = deferred<string>()
  const a = fakeWindows(() => primary.promise)
  const b = fakeWindows(() => other.promise)

  const viaPrimary = flights.run(loginFlightKey(JAR, GATEWAY), 'background', a.start)
  const viaOther = flights.run(loginFlightKey(`${JAR}-conn-work`, GATEWAY), 'background', b.start)

  expect([a.stats.opened, b.stats.opened]).toEqual([1, 1])

  // One jar failing must not put the other on hold.
  primary.reject(new Error('nope'))
  other.resolve('ok')
  await expect(viaPrimary).rejects.toThrow(/nope/)
  await expect(viaOther).resolves.toBe('ok')
  expect(flights.cooldownRemainingMs(loginFlightKey(`${JAR}-conn-work`, GATEWAY))).toBe(0)
})

test('the flight key is the jar plus the gateway origin', () => {
  expect(loginFlightKey(JAR, `${GATEWAY}/api/status`)).toBe(KEY)
  expect(loginFlightKey(JAR, 'https://other.example.test')).not.toBe(KEY)
  expect(loginFlightKey(`${JAR}-conn-x`, GATEWAY)).not.toBe(KEY)
})

// main.ts is not importable under vitest; pin which callers count as unattended.
test('refused requests and the boot cascade sign in hidden; only user actions may show a window', () => {
  const source = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
  const fetchStart = source.indexOf('function fetchJsonViaOauthSession(')
  const fetchBody = source.slice(fetchStart, source.indexOf('\n}\n', fetchStart))
  const bootStart = source.indexOf('async function waitForRemoteHermes(')
  const bootBody = source.slice(bootStart, source.indexOf('\n}\n', bootStart))

  expect(fetchBody).toContain('openOauthLoginWindow(origin, { silent: true, background: true })')
  expect(bootBody).toContain('cloudAgentSilentSignIn(remote.baseUrl, { background: true })')
  expect(source).toContain("unattended ? 'background' : 'foreground'")
})
