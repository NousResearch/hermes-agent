import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ensureMainWindow } from './main-window-lifecycle'

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

// A relaunch while the first launch is still waiting for its backend has no
// window to focus and no right to create a second one, so the extra activation
// used to be a silent no-op — the user sees nothing and clicks again. Report it
// so the caller can surface "already starting" instead of swallowing the click.
test('reports a still-starting activation instead of silently doing nothing', () => {
  const outcome = ensureMainWindow(null, {
    isReady: true,
    createWindow: () => assert.fail('a starting instance must not be duplicated'),
    focusWindow: () => assert.fail('there is no window to focus yet'),
    starting: true
  })

  assert.equal(outcome, 'starting')
})

test('classifies the ordinary outcomes for the caller', () => {
  assert.equal(
    ensureMainWindow(
      { isDestroyed: () => false },
      { isReady: true, createWindow: () => undefined, focusWindow: () => undefined }
    ),
    'focused'
  )

  assert.equal(
    ensureMainWindow(null, { isReady: true, createWindow: () => undefined, focusWindow: () => undefined }),
    'created'
  )

  assert.equal(
    ensureMainWindow(null, { isReady: false, createWindow: () => undefined, focusWindow: () => undefined }),
    'not-ready'
  )
})
