import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  activateWindow,
  decideSecondInstanceAction,
  ensureMainWindow,
  shouldQuitOnAllClosed
} from './main-window-lifecycle'

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

// Regression for #130810: closing the last window must quit on
// Windows/Linux (never strand a windowless single-instance lock holder),
// while macOS stays in the Dock unless handing off to a detached script.
test('window-all-closed quits off macOS and only on handoff for darwin', () => {
  assert.equal(shouldQuitOnAllClosed('win32', false), true)
  assert.equal(shouldQuitOnAllClosed('linux', false), true)
  assert.equal(shouldQuitOnAllClosed('darwin', false), false)
  assert.equal(shouldQuitOnAllClosed('darwin', true), true)
})

// Regression for #130810: a second launch re-creates a destroyed primary,
// defers pre-ready (whenReady boot owns the first window), and activates a
// live one instead of silently exiting.
test('second-instance routes destroyed/missing/live primaries', () => {
  assert.equal(decideSecondInstanceAction({ isDestroyed: () => true }, true), 'create')
  assert.equal(decideSecondInstanceAction(null, true), 'create')
  assert.equal(decideSecondInstanceAction(undefined, true), 'create')
  assert.equal(decideSecondInstanceAction({ isDestroyed: () => true }, false), 'defer')
  assert.equal(decideSecondInstanceAction(null, false), 'defer')
  assert.equal(decideSecondInstanceAction({ isDestroyed: () => false }, true), 'activate')
  assert.equal(decideSecondInstanceAction({ isDestroyed: () => false }, false), 'activate')
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
