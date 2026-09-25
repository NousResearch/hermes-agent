import assert from 'node:assert/strict'

import { test } from 'vitest'

import { FOCUS_GRACE_MS, windowAcceleratorAction, type WindowAcceleratorInput } from './window-accelerator'

function chord(overrides: Partial<WindowAcceleratorInput> = {}): WindowAcceleratorInput {
  return {
    alt: false,
    control: false,
    key: '',
    meta: false,
    shift: false,
    type: 'keyDown',
    ...overrides
  }
}

test('claims Close Tab on a Windows Ctrl+W keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true }), false), 'close-tab')
})

test('ignores a Windows Ctrl+W keyUp left over from another app', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: 'w', control: true }), false), 'ignore')
})

test('claims Close Tab on a macOS Cmd+W keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', meta: true }), true), 'close-tab')
})

test('ignores a macOS Cmd+W keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: 'w', meta: true }), true), 'ignore')
})

test('claims Reload on a Windows Ctrl+R keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'r', control: true }), false), 'reload')
})

test('ignores a Windows Ctrl+R keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: 'r', control: true }), false), 'ignore')
})

test('claims Reload on a macOS Cmd+R keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'r', meta: true }), true), 'reload')
})

test('ignores a macOS Cmd+R keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: 'r', meta: true }), true), 'ignore')
})

test('claims zoom reset on Ctrl+0 keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '0', control: true }), false), 'zoom-reset')
})

test('ignores a Ctrl+0 keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: '0', control: true }), false), 'ignore')
})

test('claims zoom in on Ctrl+= keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '=', control: true }), false), 'zoom-in')
})

test('claims zoom in on Ctrl++ keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '+', control: true }), false), 'zoom-in')
})

test('claims zoom in when Plus arrives as Shift+=', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '=', control: true, shift: true }), false), 'zoom-in')
  assert.equal(windowAcceleratorAction(chord({ key: '+', control: true, shift: true }), false), 'zoom-in')
})

test('ignores a Ctrl++ keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: '+', control: true }), false), 'ignore')
})

test('claims zoom out on Ctrl+- keyDown', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '-', control: true }), false), 'zoom-out')
})

test('ignores a Ctrl+- keyUp', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'keyUp', key: '-', control: true }), false), 'ignore')
})

test('ignores a char event for the same Ctrl+W chord', () => {
  assert.equal(windowAcceleratorAction(chord({ type: 'char', key: 'w', control: true }), false), 'ignore')
})

test('does not treat Ctrl+Shift+W as Close Tab', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true, shift: true }), false), 'ignore')
})

test('does not treat Ctrl+Shift+R as Reload', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'r', control: true, shift: true }), false), 'ignore')
})

test('does not treat Ctrl+Shift+0 as zoom reset', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '0', control: true, shift: true }), false), 'ignore')
})

test('does not treat Ctrl+Shift+- as zoom out', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '-', control: true, shift: true }), false), 'ignore')
})

test('does not claim a chord that uses the other platform modifier', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', meta: true }), false), 'ignore')
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true }), true), 'ignore')
})

test('does not claim an Alt-modified chord', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true, alt: true }), false), 'ignore')
  assert.equal(windowAcceleratorAction(chord({ key: '0', control: true, alt: true }), false), 'ignore')
})

test('folds letter case for Close Tab and Reload only', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'W', control: true }), false), 'close-tab')
  assert.equal(windowAcceleratorAction(chord({ key: 'R', meta: true }), true), 'reload')
})

test('swallows an auto-repeating Ctrl+W or Ctrl+R instead of acting on it (#105498)', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true, isAutoRepeat: true }), false), 'swallow')
  assert.equal(windowAcceleratorAction(chord({ key: 'w', meta: true, isAutoRepeat: true }), true), 'swallow')
  assert.equal(windowAcceleratorAction(chord({ key: 'r', control: true, isAutoRepeat: true }), false), 'swallow')
})

test('swallows Ctrl+W and Ctrl+R that land inside the post-focus grace window (#105498)', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true }), false, 0), 'swallow')
  assert.equal(windowAcceleratorAction(chord({ key: 'r', control: true }), false, FOCUS_GRACE_MS - 1), 'swallow')
})

test('acts on Ctrl+W and Ctrl+R once the post-focus grace window has passed', () => {
  assert.equal(windowAcceleratorAction(chord({ key: 'w', control: true }), false, FOCUS_GRACE_MS), 'close-tab')
  assert.equal(windowAcceleratorAction(chord({ key: 'r', meta: true }), true, FOCUS_GRACE_MS + 50), 'reload')
})

test('keeps zoom on auto-repeat and inside the focus grace window (holding Ctrl+= zooms)', () => {
  assert.equal(windowAcceleratorAction(chord({ key: '=', control: true, isAutoRepeat: true }), false), 'zoom-in')
  assert.equal(windowAcceleratorAction(chord({ key: '-', control: true }), false, 0), 'zoom-out')
})
