const assert = require('node:assert/strict')
const test = require('node:test')

const { getWindowButtonPosition, getWindowState, isWindowLive } = require('./window-state.cjs')

const FALLBACK = { x: 24, y: 6 }
const LIVE_BUTTON = { x: 99, y: 7 }

function liveWindow(overrides = {}) {
  return {
    isDestroyed: () => false,
    isFullScreen: () => false,
    getWindowButtonPosition: () => LIVE_BUTTON,
    ...overrides
  }
}

function destroyedWindow() {
  // A destroyed BrowserWindow keeps isDestroyed() working but throws on every
  // other native accessor — exactly the shape that broke #38468.
  return {
    isDestroyed: () => true,
    isFullScreen: () => {
      throw new Error('Object has been destroyed')
    },
    getWindowButtonPosition: () => {
      throw new Error('Object has been destroyed')
    }
  }
}

test('isWindowLive distinguishes null, destroyed, and live windows', () => {
  assert.equal(isWindowLive(null), false)
  assert.equal(isWindowLive(undefined), false)
  assert.equal(isWindowLive(destroyedWindow()), false)
  assert.equal(isWindowLive(liveWindow()), true)
  // A window so far gone even isDestroyed() throws must read as not-live.
  assert.equal(
    isWindowLive({
      isDestroyed: () => {
        throw new Error('Object has been destroyed')
      }
    }),
    false
  )
})

test('getWindowState does not throw for a destroyed window (regression #38468)', () => {
  const state = getWindowState(destroyedWindow(), {
    isMac: true,
    nativeOverlayWidth: 0,
    fallbackButtonPosition: FALLBACK
  })

  assert.deepEqual(state, {
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: FALLBACK
  })
})

test('getWindowState does not throw for a missing window', () => {
  const state = getWindowState(null, { isMac: false, nativeOverlayWidth: 144, fallbackButtonPosition: FALLBACK })

  assert.deepEqual(state, {
    isFullscreen: false,
    nativeOverlayWidth: 144,
    windowButtonPosition: null
  })
})

test('getWindowState reflects a live window', () => {
  const state = getWindowState(liveWindow({ isFullScreen: () => true }), {
    isMac: true,
    nativeOverlayWidth: 0,
    fallbackButtonPosition: FALLBACK
  })

  assert.deepEqual(state, {
    isFullscreen: true,
    nativeOverlayWidth: 0,
    windowButtonPosition: LIVE_BUTTON
  })
})

test('getWindowButtonPosition returns null off macOS and falls back when not live', () => {
  assert.equal(getWindowButtonPosition(liveWindow(), { isMac: false, fallbackButtonPosition: FALLBACK }), null)
  assert.equal(getWindowButtonPosition(destroyedWindow(), { isMac: true, fallbackButtonPosition: FALLBACK }), FALLBACK)
  assert.equal(getWindowButtonPosition(liveWindow(), { isMac: true, fallbackButtonPosition: FALLBACK }), LIVE_BUTTON)
})

test('getWindowButtonPosition falls back when the live window returns nothing', () => {
  const win = liveWindow({ getWindowButtonPosition: () => null })
  assert.equal(getWindowButtonPosition(win, { isMac: true, fallbackButtonPosition: FALLBACK }), FALLBACK)
})
