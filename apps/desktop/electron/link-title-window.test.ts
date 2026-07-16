import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createLinkTitleWindow,
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
  guardLinkTitleSession(
    {
      on: (e, h) => {
        handlers[e] = h
      },
      webRequest: { onBeforeRequest() {} }
    },
    value => value
  )
  handlers['will-download'](null, {
    cancel: () => {
      cancelled = true
    }
  })
  assert.ok(cancelled)
})

test('guardLinkTitleSession blocks rejected redirects and subrequests while allowing public HTTP(S)', () => {
  let beforeRequest

  const admitUrl = value => {
    const url = new URL(value)

    if (!['http:', 'https:'].includes(url.protocol) || ['10.0.0.1', '127.0.0.1'].includes(url.hostname)) {
      return null
    }

    return url.href
  }

  guardLinkTitleSession(
    {
      on() {},
      webRequest: {
        onBeforeRequest(handler) {
          beforeRequest = handler
        }
      }
    },
    admitUrl
  )

  const cancelled = (url, resourceType) => {
    let result
    beforeRequest({ resourceType, url }, value => {
      result = value.cancel
    })

    return result
  }

  assert.equal(cancelled('http://127.0.0.1/redirect', 'mainFrame'), true)
  assert.equal(cancelled('http://10.0.0.1/data', 'xhr'), true)
  assert.equal(cancelled('file:///tmp/private', 'mainFrame'), true)
  assert.equal(cancelled('https://example.com/', 'mainFrame'), false)
  assert.equal(cancelled('http://cdn.example.com/app.js', 'script'), false)
  assert.equal(cancelled('https://example.com/site.css', 'stylesheet'), true)
})

test('guardLinkTitleSession is a no-op when session.on throws', () => {
  assert.doesNotThrow(() =>
    guardLinkTitleSession(
      {
        on() {
          throw new Error()
        },
        webRequest: { onBeforeRequest() {} }
      },
      value => value
    )
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
