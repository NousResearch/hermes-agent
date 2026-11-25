import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import {
  challengePhase,
  challengeUrlAllowed,
  challengeWindowOptions,
  createChallengeWindows,
  parseChallengeRequest
} from './challenge-window'

const PORTAL = 'https://portal.nousresearch.com'
const URL_OK = `${PORTAL}/challenge?code=abc`

function makeSession() {
  return Object.assign(new EventEmitter(), {
    setPermissionRequestHandler: vi.fn(),
    setPermissionCheckHandler: vi.fn()
  })
}

function makeHarness() {
  const windows: any[] = []
  const logs: string[] = []
  // Errors the next loadURL calls reject with, in order.
  const loadErrors: Error[] = []
  const session = makeSession()

  const createWindow = (options: any) => {
    const contents = Object.assign(new EventEmitter(), {
      setAudioMuted: vi.fn(),
      setWindowOpenHandler: vi.fn()
    })

    const win = Object.assign(new EventEmitter(), {
      options,
      webContents: contents,
      shown: false,
      destroyed: false,
      loaded: '' as string,
      show() {
        this.shown = true
      },
      focus() {},
      isDestroyed() {
        return this.destroyed
      },
      destroy() {
        this.destroyed = true
      },
      loadURL(url: string) {
        this.loaded = url
        const failure = loadErrors.shift()

        return failure ? Promise.reject(failure) : Promise.resolve()
      }
    })

    windows.push(win)

    return win as any
  }

  const clock = { now: 1_000_000 }

  const challenges = createChallengeWindows({
    isReady: () => true,
    getSession: () => session as any,
    resolvePortalBaseUrl: () => PORTAL,
    createWindow,
    rememberLog: message => logs.push(message),
    now: () => clock.now
  })

  const phase = (win: any, fragment: string) =>
    win.webContents.emit('did-navigate-in-page', {}, `${URL_OK}#${fragment}`)

  return { challenges, windows, logs, session, phase, clock, loadErrors }
}

test('only <portal>/challenge URLs are ever loadable', () => {
  assert.equal(challengeUrlAllowed(URL_OK, PORTAL), true)
  assert.equal(challengeUrlAllowed(`${PORTAL}/challenge/lab`, PORTAL), true)

  for (const url of [
    'https://evil.example/challenge?code=abc',
    `${PORTAL}.evil.example/challenge`,
    `${PORTAL}/login`,
    `${PORTAL}/challenges`,
    'http://portal.nousresearch.com/challenge',
    'javascript:alert(1)',
    'file:///etc/passwd',
    ''
  ]) {
    assert.equal(challengeUrlAllowed(url, PORTAL), false, url)
  }
})

test('the phase is the URL fragment, and nothing else is a phase', () => {
  assert.equal(challengePhase(`${URL_OK}#interactive`), 'interactive')
  assert.equal(challengePhase(`${URL_OK}#done`), 'done')
  assert.equal(challengePhase(URL_OK), null)
  assert.equal(challengePhase(`${URL_OK}#constructor`), null)
})

test('the window is hidden, hardened, isolated, and not throttled', () => {
  const session = { id: 'challenge' } as any
  const options = challengeWindowOptions(session)

  assert.equal(options.show, false)
  assert.equal(options.webPreferences?.session, session)
  assert.equal(options.webPreferences?.sandbox, true)
  assert.equal(options.webPreferences?.contextIsolation, true)
  assert.equal(options.webPreferences?.nodeIntegration, false)
  assert.equal(options.webPreferences?.backgroundThrottling, false)
})

test('a pass while hidden closes the window without ever showing it', async () => {
  const { challenges, windows, phase, session } = makeHarness()
  const outcome = challenges.run({ url: URL_OK, required: true })
  const [win] = windows

  assert.equal(win.loaded, URL_OK)
  assert.equal(win.options.show, false)
  phase(win, 'working')
  phase(win, 'done')

  assert.equal(await outcome, 'done')
  assert.equal(win.shown, false)
  assert.equal(win.destroyed, true)
  // Remote content gets no permissions and no popups.
  assert.equal(session.setPermissionRequestHandler.mock.calls.length, 1)
  assert.deepEqual(win.webContents.setWindowOpenHandler.mock.calls[0][0]({ url: 'https://x.test' }), {
    action: 'deny'
  })
})

test('the page asking for the human reveals the window, and a pass lingers briefly', async () => {
  vi.useFakeTimers()

  try {
    const { challenges, windows, phase } = makeHarness()
    const outcome = challenges.run({ url: URL_OK, required: true, expiresIn: 600 })
    const [win] = windows

    phase(win, 'interactive')
    assert.equal(win.shown, true)
    // The hidden deadline no longer applies: the human has the ticket's life.
    vi.advanceTimersByTime(120_000)
    assert.equal(win.destroyed, false)

    phase(win, 'done')
    assert.equal(win.destroyed, false)
    vi.advanceTimersByTime(2_000)
    assert.equal(await outcome, 'done')
    assert.equal(win.destroyed, true)
  } finally {
    vi.useRealTimers()
  }
})

test('an optional challenge is never revealed', async () => {
  const { challenges, windows, phase } = makeHarness()
  const outcome = challenges.run({ url: URL_OK, required: false })

  phase(windows[0], 'interactive')
  assert.equal(await outcome, 'failed')
  assert.equal(windows[0].shown, false)
})

test('a hidden window that never finishes times out instead of living forever', async () => {
  vi.useFakeTimers()

  try {
    const { challenges, windows } = makeHarness()
    const outcome = challenges.run({ url: URL_OK, required: true })

    vi.advanceTimersByTime(91_000)
    assert.equal(await outcome, 'timeout')
    assert.equal(windows[0].destroyed, true)
  } finally {
    vi.useRealTimers()
  }
})

test('the page cannot take the window off the portal', () => {
  const { challenges, windows } = makeHarness()

  void challenges.run({ url: URL_OK, required: true })

  const navigate = (url: string) => {
    const event = { preventDefault: vi.fn() }

    windows[0].webContents.emit('will-navigate', event, url)

    return event.preventDefault.mock.calls.length
  }

  assert.equal(navigate(`${PORTAL}/challenge?code=abc#done`), 0)
  assert.equal(navigate('https://evil.example/'), 1)
  assert.equal(navigate('not a url'), 1)
})

test('a URL that is not a portal challenge opens nothing', async () => {
  const { challenges, windows } = makeHarness()

  assert.equal(await challenges.run({ url: 'https://evil.example/challenge', required: true }), 'refused')
  assert.equal(windows.length, 0)
})

test('two asks for the same challenge share one window', async () => {
  const { challenges, windows, phase } = makeHarness()
  const first = challenges.run({ url: URL_OK, required: true })
  const second = challenges.run({ url: URL_OK, required: true })

  assert.equal(windows.length, 1)
  phase(windows[0], 'done')
  assert.deepEqual(await Promise.all([first, second]), ['done', 'done'])
})

test('a settled challenge does not come back for the same ticket, until the memory lapses', async () => {
  const { challenges, windows, phase, clock } = makeHarness()
  const first = challenges.run({ url: URL_OK, required: true })

  phase(windows[0], 'interactive')
  windows[0].emit('closed')
  assert.equal(await first, 'closed')

  // The backend is still announcing the ticket; the window the user closed stays closed.
  assert.equal(await challenges.run({ url: URL_OK, required: true }), 'closed')
  assert.equal(windows.length, 1)

  clock.now += 11 * 60_000
  void challenges.run({ url: URL_OK, required: true })
  assert.equal(windows.length, 2)
})

test('a timed-out window may be tried again for the same ticket', async () => {
  vi.useFakeTimers()

  try {
    const { challenges, windows } = makeHarness()
    const first = challenges.run({ url: URL_OK, required: true })

    vi.advanceTimersByTime(91_000)
    assert.equal(await first, 'timeout')
    void challenges.run({ url: URL_OK, required: true })
    assert.equal(windows.length, 2)
  } finally {
    vi.useRealTimers()
  }
})

test('a server redirect off the portal is cancelled, and a foreign page cannot signal a phase', async () => {
  const { challenges, windows } = makeHarness()
  const outcome = challenges.run({ url: URL_OK, required: true })
  const [win] = windows
  const redirect = { preventDefault: vi.fn() }

  win.webContents.emit('will-redirect', redirect, 'https://evil.example/landing')
  assert.equal(redirect.preventDefault.mock.calls.length, 1)

  // A fragment on a foreign URL is not a phase: the window is not revealed.
  win.webContents.emit('did-navigate-in-page', {}, 'https://evil.example/x#interactive', true)
  assert.equal(win.shown, false)

  // A subframe cannot signal either.
  win.webContents.emit('did-navigate-in-page', {}, `${URL_OK}#interactive`, false)
  assert.equal(win.shown, false)

  // And a main-frame commit off the portal ends the run.
  win.webContents.emit('did-navigate', {}, 'https://evil.example/landing', 200)
  assert.equal(await outcome, 'failed')
  assert.equal(win.destroyed, true)
})

test('an HTTP error page ends the run instead of waiting out the deadline, and may be tried again', async () => {
  const { challenges, windows, phase } = makeHarness()
  const outcome = challenges.run({ url: URL_OK, required: true })

  windows[0].webContents.emit('did-navigate', {}, URL_OK, 503)
  assert.equal(await outcome, 'error')

  const retry = challenges.run({ url: URL_OK, required: true })
  assert.equal(windows.length, 2)
  phase(windows[1], 'done')
  assert.equal(await retry, 'done')
})

test('a superseded load is not a failed one, but a real load failure is', async () => {
  const { challenges, windows, phase, loadErrors } = makeHarness()

  loadErrors.push(Object.assign(new Error('ERR_ABORTED'), { errno: -3 }))
  const superseded = challenges.run({ url: `${URL_OK}&n=1`, required: true })

  await new Promise(resolve => setImmediate(resolve))
  assert.equal(windows[0].destroyed, false)
  phase(windows[0], 'done')
  assert.equal(await superseded, 'done')

  loadErrors.push(Object.assign(new Error('ERR_NAME_NOT_RESOLVED'), { errno: -105 }))
  assert.equal(await challenges.run({ url: `${URL_OK}&n=2`, required: true }), 'error')
})

test('an absurd ticket life cannot overflow the reveal timer', async () => {
  vi.useFakeTimers()

  try {
    const { challenges, windows, phase } = makeHarness()
    const outcome = challenges.run({ url: URL_OK, required: true, expiresIn: 1e12 })

    phase(windows[0], 'interactive')
    vi.advanceTimersByTime(60_000)
    // Still open a minute later: a 2^31 overflow would have closed it at once.
    assert.equal(windows[0].destroyed, false)
    vi.advanceTimersByTime(15 * 60_000)
    assert.equal(await outcome, 'timeout')
  } finally {
    vi.useRealTimers()
  }
})

test('the user closing a revealed window settles the run', async () => {
  const { challenges, windows, phase } = makeHarness()
  const outcome = challenges.run({ url: URL_OK, required: true })

  phase(windows[0], 'interactive')
  windows[0].emit('closed')
  assert.equal(await outcome, 'closed')
})

test('IPC payloads are parsed, not trusted', () => {
  assert.deepEqual(parseChallengeRequest({ url: URL_OK, required: false, expiresIn: 600 }), {
    url: URL_OK,
    required: false,
    expiresIn: 600
  })
  assert.deepEqual(parseChallengeRequest({ url: URL_OK }), { url: URL_OK, required: true, expiresIn: undefined })
  assert.equal(parseChallengeRequest({ url: 42 }), null)
  assert.equal(parseChallengeRequest(null), null)
  assert.equal(parseChallengeRequest({ url: 'x'.repeat(3000) }), null)
})

test('a fresh foreground attempt can reopen a closed window while status replay cannot', async () => {
  const { challenges, windows, phase } = makeHarness()
  const first = challenges.run({ url: URL_OK, required: true, attempt: 0 })

  windows[0].emit('closed')
  assert.equal(await first, 'closed')
  assert.equal(await challenges.run({ url: URL_OK, required: true, attempt: 0 }), 'closed')
  const retry = challenges.run({ url: URL_OK, required: true, attempt: 1 })

  assert.equal(windows.length, 2)
  phase(windows[1], 'done')
  assert.equal(await retry, 'done')
})

test('a required ask neither joins nor inherits an optional run of the same ticket', async () => {
  const { challenges, windows, phase } = makeHarness()
  const optional = challenges.run({ url: URL_OK, required: false })
  const required = challenges.run({ url: URL_OK, required: true })

  assert.equal(windows.length, 2)
  phase(windows[1], 'interactive')
  assert.equal(windows[1].shown, true)
  phase(windows[0], 'done')
  phase(windows[1], 'done')
  assert.equal(await optional, 'done')
  assert.equal(await required, 'done')
})

test('an optional run that failed does not answer a later required ask for the same ticket', async () => {
  const { challenges, windows, phase } = makeHarness()
  const optional = challenges.run({ url: URL_OK, required: false })

  phase(windows[0], 'interactive')
  assert.equal(await optional, 'failed')
  const required = challenges.run({ url: URL_OK, required: true })

  assert.equal(windows.length, 2)
  phase(windows[1], 'done')
  assert.equal(await required, 'done')
})
