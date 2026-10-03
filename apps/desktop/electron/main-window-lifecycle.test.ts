import assert from 'node:assert/strict'

import { test } from 'vitest'

import { activateWindow, ensureMainWindow, shouldQuitOnLastChatClosed } from './main-window-lifecycle'

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

// Regression for #130810: an explicit relaunch restores + shows + focuses
// (activation), so a minimized or tray-hidden window comes back instead of
// flashing the taskbar. This must use show(), not showInactive().
test('explicit relaunch activates a minimized tray-hidden window', () => {
  const calls: string[] = []
  let minimized = true
  let visible = false
  let focused = false

  activateWindow({
    isDestroyed: () => false,
    isMinimized: () => minimized,
    isVisible: () => visible,
    isFocused: () => focused,
    restore: () => {
      calls.push('restore')
      minimized = false
      visible = false
    },
    show: () => {
      calls.push('show')
      visible = true
    },
    focus: () => {
      calls.push('focus')
      focused = true
    }
  })

  assert.deepEqual(calls, ['restore', 'show', 'focus'])
})

// Regression for #130810: the last-chat `closed` fallback quits on
// Windows/Linux when no chat surface remains and no quit is already tearing
// down. It must not consult the overlay-suppression latch, must stay quiet
// on macOS, during handoff, with peers left, or mid-quit.
test('last-chat fallback quits only for a final non-macOS close outside a quit', () => {
  const cases: [string, boolean, number, boolean, boolean][] = [
    // platform, isQuittingForHandoff, remainingChatWindows, quitInProgress, expected
    ['win32', false, 0, false, true],
    ['linux', false, 0, false, true],
    ['darwin', false, 0, false, false],
    ['win32', true, 0, false, false],
    ['win32', false, 1, false, false],
    ['win32', false, 0, true, false]
  ]

  for (const [platform, isQuittingForHandoff, remainingChatWindows, quitInProgress, expected] of cases) {
    const input = { platform, isQuittingForHandoff, remainingChatWindows, quitInProgress }
    assert.equal(shouldQuitOnLastChatClosed(input), expected, JSON.stringify(input))
  }
})
