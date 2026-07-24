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
  assert.equal(options.x, -20000)
  assert.equal(options.y, -20000)
  assert.equal(options.skipTaskbar, true)
  assert.equal(options.focusable, false)
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

test('createLinkTitleWindow hides and zeros opacity so Windows cannot flash a blank frame', () => {
  // Regression for #64867: tier-2 title fetch for bot-walled URLs (GitHub)
  // must not paint a visible blank BrowserWindow over Desktop chat.
  const calls = { hide: 0, opacity: [] }
  const FakeBrowserWindow = function (options) {
    this.options = options
    this.webContents = { setAudioMuted() {} }
    this.hide = () => {
      calls.hide += 1
    }
    this.setOpacity = value => {
      calls.opacity.push(value)
    }
  }

  createLinkTitleWindow(FakeBrowserWindow, { id: 'link-titles' })

  assert.equal(calls.hide, 1)
  assert.deepEqual(calls.opacity, [0])
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
