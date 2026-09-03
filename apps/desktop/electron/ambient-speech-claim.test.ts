import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hudScopedSpeechOwnership } from './ambient-speech-claim'

// Renderer id 10 is the hidden main window, 42 the live HUD window.
test('with a live HUD, the hidden main renderer is denied speech ownership', () => {
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-1', senderId: 10, hudSenderId: 42 }), false)
})

test('with a live HUD, the HUD renderer owns speech even if the main window claimed first', () => {
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-1', senderId: 42, hudSenderId: 42 }), true)
})

test('speech pinning is time-independent, so claims far apart resolve identically', () => {
  // Ownership is decided by sender identity alone; nothing reads a clock, so a
  // claim arriving after the 1s dedupe interval cannot flip the outcome (#99717).
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-1', senderId: 10, hudSenderId: 42 }), false)
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-1', senderId: 42, hudSenderId: 42 }), true)
})

test('no live HUD: returns null so the first-caller-wins deduper decides speech too', () => {
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-1', senderId: 10, hudSenderId: null }), null)
})

test('non-speech cues keep the existing dedupe path even with a live HUD', () => {
  assert.equal(hudScopedSpeechOwnership({ key: 'sound:turn:s1', senderId: 10, hudSenderId: 42 }), null)
})

test('after the HUD closes, the main renderer can own new speech claims again', () => {
  // hudSenderId null models a destroyed/closed HUD window at claim time.
  assert.equal(hudScopedSpeechOwnership({ key: 'speak:msg-2', senderId: 10, hudSenderId: null }), null)
})
