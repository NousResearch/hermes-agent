import assert from 'node:assert/strict'

import { test } from 'vitest'

import { closeActionForResponse, windowCloseDecision } from './window-close-policy'

test('Windows and Linux prompt on a user window close', () => {
  assert.equal(windowCloseDecision({ platform: 'win32' }), 'prompt')
  assert.equal(windowCloseDecision({ platform: 'linux' }), 'prompt')
})

test('macOS and real application quits bypass the close prompt', () => {
  assert.equal(windowCloseDecision({ platform: 'darwin' }), 'proceed')
  assert.equal(windowCloseDecision({ platform: 'win32', quitInProgress: true }), 'proceed')
  assert.equal(windowCloseDecision({ handoffInProgress: true, platform: 'linux' }), 'proceed')
})

test('a pending close or active-work prompt holds repeated close gestures', () => {
  assert.equal(windowCloseDecision({ platform: 'win32', promptOpen: true }), 'hold')
  assert.equal(windowCloseDecision({ platform: 'linux', promptOpen: true }), 'hold')
})

test('close dialog responses map to minimize, quit, and cancel', () => {
  assert.equal(closeActionForResponse(0), 'minimize')
  assert.equal(closeActionForResponse(1), 'quit')
  assert.equal(closeActionForResponse(2), 'cancel')
  assert.equal(closeActionForResponse(-1), 'cancel')
})
