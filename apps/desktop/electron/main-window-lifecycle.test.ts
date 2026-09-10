import assert from 'node:assert/strict'

import { test } from 'vitest'

import { closeWindowsForDrain, ensureMainWindow, shouldQuitAfterWindowAllClosed } from './main-window-lifecycle'

test('closing the last desktop window quits on every platform', () => {
  assert.equal(shouldQuitAfterWindowAllClosed(), true)
})

test('destroying renderer windows for a deferred drain is idempotent', () => {
  let liveDestroyed = 0
  let alreadyDestroyed = 0
  const windows = [
    { destroy: () => void (liveDestroyed += 1), isDestroyed: () => false },
    { destroy: () => void (alreadyDestroyed += 1), isDestroyed: () => true }
  ]

  assert.equal(closeWindowsForDrain(windows), 1)
  assert.equal(liveDestroyed, 1)
  assert.equal(alreadyDestroyed, 0)
})

test('recreates a destroyed primary window without focusing it', () => {
  const destroyedWindow = {
    isDestroyed: () => true
  }

  let createCalls = 0
  let focusCalls = 0

  ensureMainWindow(destroyedWindow, {
    isReady: true,
    createWindow: () => {
      createCalls += 1
    },
    focusWindow: () => {
      focusCalls += 1
    }
  })

  assert.equal(createCalls, 1)
  assert.equal(focusCalls, 0)
})

test('waits for app readiness before recreating a primary window', () => {
  let createCalls = 0

  ensureMainWindow(null, {
    isReady: false,
    createWindow: () => {
      createCalls += 1
    },
    focusWindow: () => assert.fail('missing window must not be focused')
  })

  assert.equal(createCalls, 0)
})

test('focuses a live primary window for a normal second launch', () => {
  const liveWindow = {
    isDestroyed: () => false
  }

  let focusedWindow = null

  ensureMainWindow(liveWindow, {
    isReady: true,
    createWindow: () => assert.fail('live window must not be replaced'),
    focusWindow: window => {
      focusedWindow = window
    }
  })

  assert.equal(focusedWindow, liveWindow)
})

test('leaves live-window focus to deep-link delivery', () => {
  const liveWindow = {
    isDestroyed: () => false
  }

  ensureMainWindow(liveWindow, {
    isReady: true,
    createWindow: () => assert.fail('live window must not be replaced'),
    focusWindow: () => assert.fail('deep-link delivery owns focus'),
    focusExisting: false
  })
})
