import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createLinkTitleWindow,
  decideLinkTitleRequest,
  guardLinkTitleSession,
  linkTitleWindowOptions,
  readLinkTitleWindowTitle
} from './link-title-window'

function makeFakeBrowserWindow() {
  const calls = { audioMuted: [] }

  const FakeBrowserWindow = function (options) {
    this.options = options
    this.webContents = {
      setAudioMuted(value) {
        calls.audioMuted.push(value)
      }
    }
  }

  return { FakeBrowserWindow, calls }
}

test('linkTitleWindowOptions keeps the offscreen, hardened defaults', () => {
  const session = { id: 'link-titles' }
  const options = linkTitleWindowOptions(session)

  assert.equal(options.show, false)
  assert.equal(options.webPreferences.session, session)
  assert.equal(options.webPreferences.contextIsolation, true)
  assert.equal(options.webPreferences.sandbox, true)
  assert.equal(options.webPreferences.nodeIntegration, false)
})

test('createLinkTitleWindow mutes audio so historical links never autoplay sound', () => {
  // Regression for #49505: the hidden title-fetch window loaded YouTube/watch
  // URLs (to read their <title>) without muting, leaking ~2s of audio on every
  // history re-render.
  const { FakeBrowserWindow, calls } = makeFakeBrowserWindow()

  const window = createLinkTitleWindow(FakeBrowserWindow, { id: 'link-titles' })

  assert.ok(window instanceof FakeBrowserWindow)
  assert.deepEqual(calls.audioMuted, [true])
})

test('createLinkTitleWindow still returns the window if muting throws', () => {
  const ThrowingBrowserWindow = function (options) {
    this.options = options
    this.webContents = {
      setAudioMuted() {
        throw new Error('webContents unavailable')
      }
    }
  }

  const window = createLinkTitleWindow(ThrowingBrowserWindow, { id: 'link-titles' })

  assert.ok(window instanceof ThrowingBrowserWindow)
})

test('guardLinkTitleSession cancels downloads triggered by the title-fetch window', () => {
  let cancelled = false
  const handlers = {}
  guardLinkTitleSession({
    on: (e, h) => {
      handlers[e] = h
    }
  })
  handlers['will-download'](null, {
    cancel: () => {
      cancelled = true
    }
  })
  assert.ok(cancelled)
})

test('guardLinkTitleSession is a no-op when session.on throws', () => {
  assert.doesNotThrow(() =>
    guardLinkTitleSession({
      on() {
        throw new Error()
      }
    })
  )
})

test('readLinkTitleWindowTitle returns empty for missing or destroyed windows', () => {
  assert.equal(readLinkTitleWindowTitle(null), '')
  assert.equal(readLinkTitleWindowTitle(undefined), '')
  assert.equal(readLinkTitleWindowTitle({ isDestroyed: () => true }), '')
})

test('readLinkTitleWindowTitle returns empty when webContents is destroyed', () => {
  const window = {
    isDestroyed: () => false,
    webContents: { isDestroyed: () => true, getTitle: () => 'Should Not Read' }
  }

  assert.equal(readLinkTitleWindowTitle(window), '')
})

test('readLinkTitleWindowTitle swallows getTitle throws after teardown', () => {
  const window = {
    isDestroyed: () => false,
    webContents: {
      isDestroyed: () => false,
      getTitle: () => {
        throw new Error('Object has been destroyed')
      }
    }
  }

  assert.equal(readLinkTitleWindowTitle(window), '')
})

test('readLinkTitleWindowTitle returns trimmed page title', () => {
  const window = {
    isDestroyed: () => false,
    webContents: {
      isDestroyed: () => false,
      getTitle: () => 'Example Domain'
    }
  }

  assert.equal(readLinkTitleWindowTitle(window), 'Example Domain')
})

// ─── Review B6: the hidden BrowserWindow follows redirects itself ────────────
// Tier 1's per-hop curl guard (review B5) vetted only the URLs *we* fetch. When
// tier 1 misses, runRenderTitleJob loads the original URL in a hidden window
// and CHROMIUM walks the redirect chain — every hop a fresh request the old
// will-download-only guard never looked at. The same 30x→loopback/RFC1918
// chain B5 closed re-opens through the fallback leg. The fix gates EVERY
// request the window makes through the tier-1 verdict (hostname guard + our
// own DNS resolution), one chokepoint that needs no details.ip timing.

function makeGuardedSession(io) {
  const handlers = { events: {}, beforeRequest: null }

  const session = {
    on: (event, handler) => {
      handlers.events[event] = handler
    },
    webRequest: {
      onBeforeRequest: (filterOrHandler, maybeHandler) => {
        // Electron allows (handler) or (filter, handler); the guard uses the
        // one-argument form.
        handlers.beforeRequest = maybeHandler || filterOrHandler
      }
    }
  }

  guardLinkTitleSession(session, io)

  return { session, handlers }
}

test('decideLinkTitleRequest refuses private hostname literals on any leg', () => {
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://127.0.0.1/admin'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://169.254.169.254/latest/meta-data/'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://10.0.0.5/router'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://[::1]/'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://192.168.1.1/'), true)
})

test('decideLinkTitleRequest refuses localhost-shaped and single-label names', () => {
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://localhost:8080/'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://intranet/'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://box.internal/'), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'http://printer.local/'), true)
})

test('decideLinkTitleRequest refuses names whose DNS answers private or empty', () => {
  // The SSRF half: a fresh attacker-controlled name that answers with an
  // RFC1918/loopback address, and unresolvable names (deny, not allow).
  assert.equal(decideLinkTitleRequest('mainFrame', 'https://hop.example/x', { resolvedAddresses: ['10.1.2.3'] }), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'https://hop.example/x', { resolvedAddresses: ['192.168.0.9'] }), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'https://hop.example/x', { resolvedAddresses: ['::ffff:127.0.0.1'] }), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'https://hop.example/x', { resolvedAddresses: [] }), true)
})

test('decideLinkTitleRequest allows a public name resolving public', () => {
  assert.equal(decideLinkTitleRequest('mainFrame', 'https://example.com/page', { resolvedAddresses: ['93.184.216.34'] }), false)
})

test('decideLinkTitleRequest keeps the resource-type blocks and denies junk URLs', () => {
  assert.equal(decideLinkTitleRequest('imageset', 'https://example.com/x.png', { resolvedAddresses: ['93.184.216.34'] }), true)
  assert.equal(decideLinkTitleRequest('stylesheet', 'https://example.com/x.css', { resolvedAddresses: ['93.184.216.34'] }), true)
  assert.equal(decideLinkTitleRequest('mainFrame', 'not a url'), true)
})

test('decideLinkTitleRequest refuses when the connected IP is private even if DNS said public', () => {
  // details.ip, when Chromium supplies it, outranks our resolution.
  assert.equal(
    decideLinkTitleRequest('mainFrame', 'https://hop.example/x', {
      connectedIp: '127.0.0.1',
      resolvedAddresses: ['93.184.216.34']
    }),
    true
  )
})

test('guardLinkTitleSession cancels a redirect hop to a private literal before resolving', () => {
  let resolved = 0
  const { handlers } = makeGuardedSession({ resolveHost: () => (resolved += 1) })

  const decisions = []
  handlers.beforeRequest({ resourceType: 'mainFrame', url: 'http://127.0.0.1/secret' }, decision => decisions.push(decision))

  assert.deepEqual(decisions, [{ cancel: true }])
  assert.equal(resolved, 0)
})

test('guardLinkTitleSession applies the DNS verdict before allowing a public-named request', async () => {
  const resolveHostCalls = []

  const { handlers } = makeGuardedSession({
    resolveHost: hostname => {
      resolveHostCalls.push(hostname)

      return Promise.resolve(['10.0.0.1'])
    }
  })

  const decided = new Promise(resolve => {
    handlers.beforeRequest({ resourceType: 'mainFrame', url: 'https://hop.example/one' }, resolve)
  })

  assert.deepEqual(await decided, { cancel: true })
  assert.deepEqual(resolveHostCalls, ['hop.example'])
})

test('guardLinkTitleSession lets a public-resolving request through and keeps resource blocks', async () => {
  const { handlers } = makeGuardedSession({ resolveHost: () => Promise.resolve(['93.184.216.34']) })

  const mainFrame = new Promise(resolve => {
    handlers.beforeRequest({ resourceType: 'mainFrame', url: 'https://example.com/' }, resolve)
  })

  const image = new Promise(resolve => {
    handlers.beforeRequest({ resourceType: 'imageset', url: 'https://example.com/logo.png' }, resolve)
  })

  assert.deepEqual(await mainFrame, { cancel: false })
  assert.deepEqual(await image, { cancel: true })
})

test('guardLinkTitleSession resolves a failing resolver to deny, never allow', async () => {
  const { handlers } = makeGuardedSession({ resolveHost: () => Promise.reject(new Error('dns down')) })

  const decided = new Promise(resolve => {
    handlers.beforeRequest({ resourceType: 'mainFrame', url: 'https://example.com/' }, resolve)
  })

  assert.deepEqual(await decided, { cancel: true })
})

test('guardLinkTitleSession still cancels downloads alongside the request guard', () => {
  let cancelled = false
  const { handlers } = makeGuardedSession({ resolveHost: () => Promise.resolve(['93.184.216.34']) })

  handlers.events['will-download'](null, {
    cancel: () => {
      cancelled = true
    }
  })

  assert.ok(cancelled)
})

test('guardLinkTitleSession is a no-op when session methods throw', () => {
  assert.doesNotThrow(() =>
    guardLinkTitleSession({
      on() {
        throw new Error()
      },
      webRequest: {
        onBeforeRequest() {
          throw new Error()
        }
      }
    })
  )
})
