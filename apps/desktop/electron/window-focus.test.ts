/**
 * Behavior contracts for the pure window-focus sequencing: guards, ordering
 * (restore → show → focus → activate), and the macOS application-activation
 * step that actually raises the app when another application is active.
 */

import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { type FocusableWindow, focusTargetWindow } from './window-focus'

function fakeWindow(overrides: Partial<FocusableWindow> = {}) {
  const calls: string[] = []

  const base: FocusableWindow = {
    isDestroyed: vi.fn(() => false),
    isMinimized: vi.fn(() => false),
    restore: vi.fn(() => calls.push('restore')),
    isVisible: vi.fn(() => true),
    show: vi.fn(() => calls.push('show')),
    focus: vi.fn(() => calls.push('focus'))
  }

  const win: FocusableWindow = { ...base, ...overrides }

  return { win, calls }
}

test('null or destroyed window is a no-op and never activates the app', () => {
  const activate = vi.fn()
  assert.doesNotThrow(() => focusTargetWindow(null, activate))
  assert.doesNotThrow(() => focusTargetWindow(undefined, activate))

  const destroyed = fakeWindow({ isDestroyed: () => true })
  focusTargetWindow(destroyed.win, activate)
  assert.equal(activate.mock.calls.length, 0)
  assert.equal(destroyed.calls.length, 0)
})

test('restores a minimized window before focusing', () => {
  const { win, calls } = fakeWindow({ isMinimized: () => true })
  focusTargetWindow(win)
  assert.deepEqual(calls, ['restore', 'focus'])
})

test('shows a hidden window before focusing', () => {
  const { win, calls } = fakeWindow({ isVisible: () => false })
  focusTargetWindow(win)
  assert.deepEqual(calls, ['show', 'focus'])
})

test('an already-visible window is only focused', () => {
  const { win, calls } = fakeWindow()
  focusTargetWindow(win)
  assert.deepEqual(calls, ['focus'])
})

test('focus ordering: restore → show → focus → activate application', () => {
  const order: string[] = []

  const win: FocusableWindow = {
    isDestroyed: () => false,
    isMinimized: () => true,
    restore: () => order.push('restore'),
    isVisible: () => false,
    show: () => order.push('show'),
    focus: () => order.push('focus')
  }

  const activate = () => order.push('activate')

  focusTargetWindow(win, activate)
  assert.deepEqual(order, ['restore', 'show', 'focus', 'activate'])
})

test('without an activation callback, focusing still works (non-mac callers)', () => {
  const { win, calls } = fakeWindow()
  focusTargetWindow(win)
  assert.deepEqual(calls, ['focus'])
})

test('activation runs after the window is key, even when already visible', () => {
  const { win } = fakeWindow()
  const activate = vi.fn()
  focusTargetWindow(win, activate)
  assert.equal(activate.mock.calls.length, 1)
})
