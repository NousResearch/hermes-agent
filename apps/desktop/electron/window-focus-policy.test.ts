/**
 * Invariants for the Windows foreground-steal fix (#83998).
 *
 * `focusWindow()` runs for every ambient window raise (tray restore,
 * notification click, second-instance, dock/taskbar activate). On Windows,
 * `show()` + unconditional `focus()` seizes the OS foreground and dismisses
 * other apps' native dialogs while Hermes streams in the background. The
 * main-process call sites now follow the policy in window-focus-policy.ts;
 * these tests pin it.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { revealAction, shouldFocusToTakeKeyboard } from './window-focus-policy'

test('an invisible window is revealed without stealing foreground (showInactive)', () => {
  assert.equal(revealAction(false), 'showInactive')
})

test('an already-visible window is left alone by the reveal path', () => {
  assert.equal(revealAction(true), 'none')
})

test('focus is only taken when the window does not already have it', () => {
  assert.equal(shouldFocusToTakeKeyboard({ isFocused: () => false }), true)
})

test('an already-focused window never pumps the OS foreground path', () => {
  // The #83998 regression: a redundant .focus() on a focused window still
  // runs SetForegroundWindow on Windows, dismissing another app's dialog.
  assert.equal(shouldFocusToTakeKeyboard({ isFocused: () => true }), false)
})

// Native BrowserWindow methods validate their receiver; detached methods fail
// even when the window is still alive. Exercise both focus states.
test('focus policy preserves the native window receiver', () => {
  const window = {
    focused: false,
    isFocused() {
      assert.equal(this, window, 'BrowserWindow method must retain its receiver')

      return this.focused
    }
  }

  assert.equal(shouldFocusToTakeKeyboard(window), true)
  window.focused = true
  assert.equal(shouldFocusToTakeKeyboard(window), false)
})
