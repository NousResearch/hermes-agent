import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { OauthLoginFlights } from './oauth-login-flights'
import { type OauthLoginWindowLike, startOauthLoginWindow } from './oauth-login-window'

const GATEWAY = 'https://fedora.example.ts.net'

/** A queued load outcome that never settles: an edge holding the request open. */
const HANG = Symbol('hang')

/** A login window double that records visibility and lets a test fire events.
 *  Each load rejects with the next queued failure (at once, or `{ after, error }` later; HANG
 *  never settles), then succeeds. */
function fakeWindow(loadFailures: unknown[] = []) {
  const closed: Array<() => void> = []
  const navigations: Array<(event: unknown, url: string) => void> = []

  const state = { destroyed: false, focused: 0, loaded: '', loads: 0, shown: false, visible: false }

  const win: OauthLoginWindowLike = {
    destroy: () => {
      state.destroyed = true
    },
    focus: () => {
      state.focused += 1
    },
    hide: () => {
      state.visible = false
    },
    isDestroyed: () => state.destroyed,
    isVisible: () => state.visible,
    loadURL: async url => {
      state.loaded = url
      state.loads += 1

      if (loadFailures.length > 0) {
        const failure = loadFailures.shift()

        if (failure === HANG) {
          return new Promise(() => {})
        }

        if (failure && typeof failure === 'object' && 'after' in failure) {
          const { after, error } = failure as { after: number; error: Error }

          return new Promise((_resolve, reject) => setTimeout(() => reject(error), after))
        }

        throw failure
      }
    },
    on: (_event, listener) => closed.push(listener),
    show: () => {
      state.shown = true
      state.visible = true
    },
    webContents: {
      on: (event, listener) => {
        if (event === 'did-navigate') {
          navigations.push(listener)
        }
      }
    }
  }

  return {
    close: () => closed.forEach(listener => listener()),
    navigate: (url = `${GATEWAY}/`) => navigations.forEach(listener => listener({}, url)),
    state,
    win
  }
}

function harness(cookie = { present: false }, loadFailures: unknown[] = []) {
  const windows: Array<ReturnType<typeof fakeWindow>> = []
  const created: Array<{ show: boolean; title: string }> = []

  const deps = {
    createWindow: (options: { session: string; show: boolean; title: string }) => {
      const handle = fakeWindow(loadFailures)
      handle.state.visible = options.show
      windows.push(handle)
      created.push({ show: options.show, title: options.title })

      return handle.win
    },
    hasSessionCookie: async () => cookie.present,
    isExpectedNavigationAbort: () => false,
    loadOptionsFor: () => ({}),
    log: () => {},
    onLanded: vi.fn()
  }

  const start = (request: { silent: boolean; background: boolean }) =>
    startOauthLoginWindow({ baseUrl: GATEWAY, session: 'jar', ...request }, deps)

  return { cookie, created, deps, start, windows }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

test('an unattended sign-in never shows its window and gives up after the deadline', async () => {
  const h = harness()
  const attempt = h.start({ silent: true, background: true })
  const outcome = attempt.result.catch(error => error)

  expect(h.created).toEqual([{ show: false, title: 'Connecting to Hermes Cloud agent…' }])
  expect(h.windows[0].state.loaded).toBe(`${GATEWAY}/`)

  // Past the 2.5s reveal point a foreground silent window would have shown.
  await vi.advanceTimersByTimeAsync(11_999)
  expect(h.windows[0].state.shown).toBe(false)

  await vi.advanceTimersByTimeAsync(1)
  expect(await outcome).toBeInstanceOf(Error)
  expect(h.windows[0].state).toMatchObject({ destroyed: true, shown: false })
})

test('an unattended attempt that lands on the gateway chooser stops at once', async () => {
  const h = harness()
  const attempt = h.start({ silent: true, background: true })

  h.windows[0].navigate(`${GATEWAY}/login?next=%2F`)
  await expect(attempt.result).rejects.toThrow(/sign-in required/)
  expect(h.windows[0].state).toMatchObject({ destroyed: true, shown: false })
})

test('a revealed attempt stays open on the chooser for the user', async () => {
  const h = harness()
  const attempt = h.start({ silent: true, background: true })
  let settled = false

  void attempt.result.then(
    () => (settled = true),
    () => (settled = true)
  )
  attempt.reveal()
  h.windows[0].navigate(`${GATEWAY}/login`)
  await vi.advanceTimersByTimeAsync(1_000)
  expect(settled).toBe(false)
})

test('a click that joins while the chooser check is pending keeps its window', async () => {
  const h = harness()

  let resolveCookie: (present: boolean) => void = () => {}

  h.deps.hasSessionCookie = () =>
    new Promise<boolean>(resolve => {
      resolveCookie = resolve
    })

  const attempt = h.start({ silent: true, background: true })
  let outcome = 'pending'

  void attempt.result.then(
    () => (outcome = 'resolved'),
    error => (outcome = `rejected: ${error.message}`)
  )

  h.windows[0].navigate(`${GATEWAY}/login`)
  attempt.reveal()
  resolveCookie(false)
  await vi.advanceTimersByTimeAsync(0)

  expect(h.windows[0].state).toMatchObject({ destroyed: false, shown: true })
  expect(outcome).toBe('pending')
})

test('a foreground silent sign-in reveals itself when auto-SSO does not finish', async () => {
  const h = harness()

  h.start({ silent: true, background: false })
  expect(h.created[0].show).toBe(false)

  await vi.advanceTimersByTimeAsync(2_500)
  expect(h.windows[0].state.shown).toBe(true)
})

const reset = () => new Error("ERR_CONNECTION_RESET (-101) loading 'https://fedora.example.ts.net/'")

test('a sleeping agent that resets the first loads is reloaded, hidden, until it answers', async () => {
  const h = harness({ present: false }, [reset(), reset(), reset()])
  const attempt = h.start({ silent: true, background: false })

  await vi.advanceTimersByTimeAsync(3_000)
  expect(h.windows[0].state).toMatchObject({ destroyed: false, loads: 4, shown: false })

  await vi.advanceTimersByTimeAsync(2_500)
  expect(h.windows[0].state.shown).toBe(true)

  h.cookie.present = true
  h.windows[0].navigate()
  await expect(attempt.result).resolves.toEqual({ baseUrl: GATEWAY, ok: true })
})

test('an agent that never wakes fails once the wake budget runs out', async () => {
  const h = harness({ present: false }, Array.from({ length: 100 }, reset))
  const attempt = h.start({ silent: true, background: false })
  const outcome = attempt.result.catch(error => error)

  await vi.advanceTimersByTimeAsync(55_000)
  expect(h.windows[0].state.destroyed).toBe(false)

  await vi.advanceTimersByTimeAsync(6_000)
  expect(await outcome).toMatchObject({ message: expect.stringMatching(/ERR_CONNECTION_RESET/) })
  expect(h.windows[0].state.destroyed).toBe(true)
})

test('a waking agent whose retry hangs still fails once the wake budget runs out', async () => {
  const h = harness({ present: false }, [reset(), HANG])
  const outcome = h.start({ silent: true, background: false }).result.catch(error => error)

  await vi.advanceTimersByTimeAsync(59_000)
  expect(h.windows[0].state).toMatchObject({ destroyed: false, loads: 2, shown: false })

  await vi.advanceTimersByTimeAsync(1_000)
  expect(await outcome).toMatchObject({ message: expect.stringMatching(/ERR_CONNECTION_RESET/) })
  expect(h.windows[0].state.destroyed).toBe(true)
})

test('an agent that answered is not cut off by the wake budget while the user signs in', async () => {
  const h = harness({ present: false }, [reset()])
  const attempt = h.start({ silent: true, background: false })

  await vi.advanceTimersByTimeAsync(120_000)
  expect(h.windows[0].state).toMatchObject({ destroyed: false, shown: true })

  h.cookie.present = true
  h.windows[0].navigate()
  await expect(attempt.result).resolves.toEqual({ baseUrl: GATEWAY, ok: true })
})

test('a window revealed before the first reset hides while the agent wakes', async () => {
  // The edge holds the first connection past the 2.5 s reveal, then resets it.
  const h = harness({ present: false }, [{ after: 3_000, error: reset() }, reset()])
  const attempt = h.start({ silent: true, background: false })

  await vi.advanceTimersByTimeAsync(2_500)
  expect(h.windows[0].state.visible).toBe(true)

  await vi.advanceTimersByTimeAsync(500)
  expect(h.windows[0].state).toMatchObject({ loads: 1, visible: false })

  await vi.advanceTimersByTimeAsync(2_000)
  expect(h.windows[0].state).toMatchObject({ loads: 3, visible: false })

  await vi.advanceTimersByTimeAsync(2_500)
  expect(h.windows[0].state.visible).toBe(true)

  h.cookie.present = true
  h.windows[0].navigate()
  await expect(attempt.result).resolves.toEqual({ baseUrl: GATEWAY, ok: true })
})

test('a background sign-in does not wait for a waking gateway', async () => {
  const h = harness({ present: false }, [reset()])
  const outcome = h.start({ silent: true, background: true }).result.catch(error => error)

  await vi.advanceTimersByTimeAsync(0)
  expect(await outcome).toMatchObject({ message: expect.stringMatching(/ERR_CONNECTION_RESET/) })
  expect(h.windows[0].state.loads).toBe(1)
})

test('an interactive sign-in to a gateway that refuses the connection fails at once', async () => {
  const h = harness({ present: false }, [new Error('ERR_CONNECTION_REFUSED (-102)')])
  const outcome = h.start({ silent: false, background: false }).result.catch(error => error)

  await vi.advanceTimersByTimeAsync(0)
  expect(await outcome).toMatchObject({ message: expect.stringMatching(/ERR_CONNECTION_REFUSED/) })
  expect(h.windows[0].state.loads).toBe(1)
})

test('an interactive sign-in opens visible on the provider chooser', () => {
  const h = harness()

  h.start({ silent: false, background: false })
  expect(h.created).toEqual([{ show: true, title: 'Sign in to Hermes gateway' }])
  expect(h.windows[0].state.loaded).toBe(`${GATEWAY}/login`)
})

test('reveal promotes a hidden attempt and lifts its unattended deadline', async () => {
  const h = harness()
  const attempt = h.start({ silent: true, background: true })
  let settled = false

  void attempt.result.then(
    () => (settled = true),
    () => (settled = true)
  )

  attempt.reveal()
  expect(h.windows[0].state).toMatchObject({ focused: 1, shown: true })

  await vi.advanceTimersByTimeAsync(60_000)
  expect(settled).toBe(false)

  h.cookie.present = true
  h.windows[0].navigate()
  await expect(attempt.result).resolves.toEqual({ baseUrl: GATEWAY, ok: true })
  expect(h.deps.onLanded).toHaveBeenCalledTimes(1)
})

test('closing the window rejects the attempt', async () => {
  const h = harness()
  const attempt = h.start({ silent: false, background: false })

  h.windows[0].close()
  await expect(attempt.result).rejects.toThrow(/closed before authentication/)
})

test('overnight burst: ten refused requests open one hidden window and none is ever shown', async () => {
  const h = harness()
  const flights = new OauthLoginFlights({ joinTimeoutMs: 15_000 })
  const key = `jar ${GATEWAY}`

  const burst = Array.from({ length: 10 }, () =>
    flights.run(key, 'background', () => h.start({ silent: true, background: true })).catch(error => error)
  )

  await vi.advanceTimersByTimeAsync(12_000)
  const outcomes = await Promise.all(burst)

  expect(outcomes.every(outcome => outcome instanceof Error)).toBe(true)
  expect(h.windows).toHaveLength(1)
  expect(h.windows[0].state.shown).toBe(false)

  // The next poll right after is held by the backoff instead of opening another.
  await expect(flights.run(key, 'background', () => h.start({ silent: true, background: true }))).rejects.toThrow(
    /held/
  )
  expect(h.windows).toHaveLength(1)

  // The user then clicks Sign in: one visible window, never held.
  void flights.run(key, 'foreground', () => h.start({ silent: false, background: false }))
  expect(h.windows).toHaveLength(2)
  expect(h.windows[1].state.visible).toBe(true)
})
