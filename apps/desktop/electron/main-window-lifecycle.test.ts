import assert from 'node:assert/strict'

import { test } from 'vitest'

import { deliverDeepLink, ensureMainWindow } from './main-window-lifecycle'

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

test('delivers a deep link after restoring and revealing a hidden minimized window', () => {
  const calls: string[] = []
  const payload = { kind: 'blueprint', name: 'morning-brief', params: { time: '08:00' } }

  const window = {
    focus: () => calls.push('focus'),
    isDestroyed: () => false,
    isMinimized: () => true,
    isVisible: () => false,
    restore: () => calls.push('restore'),
    show: () => calls.push('show'),
    webContents: {
      send: (channel: string, value: unknown) => calls.push(`send:${channel}:${JSON.stringify(value)}`)
    }
  }

  const delivered = deliverDeepLink(window, payload)

  assert.equal(delivered, true)
  assert.deepEqual(calls, ['restore', 'show', 'focus', `send:hermes:deep-link:${JSON.stringify(payload)}`])
})

test('does not deliver a deep link to a destroyed window', () => {
  const window = {
    focus: () => assert.fail('destroyed window must not be focused'),
    isDestroyed: () => true,
    isMinimized: () => false,
    isVisible: () => false,
    restore: () => assert.fail('destroyed window must not be restored'),
    show: () => assert.fail('destroyed window must not be shown'),
    webContents: {
      send: () => assert.fail('destroyed window must not receive a deep link')
    }
  }

  const delivered = deliverDeepLink(window, { kind: 'blueprint', name: 'brief', params: {} })

  assert.equal(delivered, false)
})
