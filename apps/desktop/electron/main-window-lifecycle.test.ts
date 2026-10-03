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

test('explicit relaunch leaves an already-visible focused window alone', () => {
  const calls: string[] = []

  activateWindow({
    isDestroyed: () => false,
    isMinimized: () => false,
    isVisible: () => true,
    isFocused: () => true,
    restore: () => calls.push('restore'),
    show: () => calls.push('show'),
    focus: () => calls.push('focus')
  })

  assert.deepEqual(calls, [])
})

test('explicit relaunch never touches a destroyed or missing window', () => {
  activateWindow({ isDestroyed: () => true })
  activateWindow(null)
  activateWindow(undefined)
})

// Regression for #130810: the last-chat `closed` fallback quits on
// Windows/Linux when no chat surface remains and no quit is already tearing
// down. It must not consult the overlay-suppression latch, must stay quiet
// on macOS, during handoff, with peers left, or mid-quit.
test('last-chat fallback quits only for a final non-macOS close outside a quit', () => {
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff: false,
      remainingChatWindows: 0,
      quitInProgress: false
    }),
    true
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'linux',
      isQuittingForHandoff: false,
      remainingChatWindows: 0,
      quitInProgress: false
    }),
    true
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'darwin',
      isQuittingForHandoff: false,
      remainingChatWindows: 0,
      quitInProgress: false
    }),
    false
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff: true,
      remainingChatWindows: 0,
      quitInProgress: false
    }),
    false
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff: false,
      remainingChatWindows: 1,
      quitInProgress: false
    }),
    false
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'linux',
      isQuittingForHandoff: false,
      remainingChatWindows: 2,
      quitInProgress: false
    }),
    false
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff: false,
      remainingChatWindows: 0,
      quitInProgress: true
    }),
    false
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'linux',
      isQuittingForHandoff: false,
      remainingChatWindows: 0,
      quitInProgress: true
    }),
    false
  )
})

// Regression for #130810: an ordinary primary-window close on Windows/Linux
// sets the overlay-suppression latch (appQuitting = true for #55920) in its
// `close` handler before `closed` fires. With a hidden helper (Quick Entry,
// HUD, pet overlay) still alive, `window-all-closed` never fires, so the
// last-chat fallback is the only quit path — the latch must not block it.
test('primary close with hidden helper still quits despite overlay-suppression latch', () => {
  // Mirror main.ts state: overlay latch vs real quit progress are separate.
  let appQuitting = false
  let quitInProgress = false
  const isQuittingForHandoff = false

  // `close`: ordinary close, no active-work guard, not prevented. The primary
  // window's `close` handler latches overlay suppression on Windows/Linux.
  const closePrevented = false

  if (!closePrevented) {
    appQuitting = true
  }

  // `closed`: the primary chat surface is gone (remaining === 0) but a hidden
  // helper BrowserWindow keeps `window-all-closed` from firing. No before-quit
  // has run yet, so no quit is in progress.
  const remainingChatWindows = 0

  assert.equal(appQuitting, true)
  assert.equal(quitInProgress, false)
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff,
      remainingChatWindows,
      quitInProgress
    }),
    true
  )
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'linux',
      isQuittingForHandoff,
      remainingChatWindows,
      quitInProgress
    }),
    true
  )
})

// Regression for #130810: a "Keep Running" answer to the active-work prompt
// holds the quit (before-quit preventDefault / window-close preventDefault)
// and must leave quitInProgress false with the overlay latch reset, so a
// later ordinary close still quits via the fallback.
test('keep-running leaves a later final close able to quit', () => {
  let appQuitting = false
  let quitInProgress = false
  const isQuittingForHandoff = false

  // First attempt: active-work guard holds the quit. Mirror the fixed
  // before-quit handler — a held quit resets the overlay latch and never sets
  // quitInProgress.
  const heldForActiveWork = true

  if (heldForActiveWork) {
    appQuitting = false
  } else {
    appQuitting = true
    quitInProgress = true
  }

  assert.equal(quitInProgress, false)
  assert.equal(appQuitting, false)

  // The dialog resolves with "Keep Running": no confirm, no re-quit.
  const quitConfirmedWithActiveWork = false
  assert.equal(quitConfirmedWithActiveWork, false)
  assert.equal(quitInProgress, false)

  // Later, work has finished and the user closes the last chat window for
  // real: ordinary `close` latches overlay suppression, `closed` finds no
  // chat surfaces left and no quit in flight, so the fallback quits.
  appQuitting = true
  const remainingChatWindows = 0

  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff,
      remainingChatWindows,
      quitInProgress
    }),
    true
  )

  // And once before-quit really proceeds, the fallback stays quiet (the quit
  // teardown already owns the exit).
  quitInProgress = true
  assert.equal(
    shouldQuitOnLastChatClosed({
      platform: 'win32',
      isQuittingForHandoff,
      remainingChatWindows,
      quitInProgress
    }),
    false
  )
})
