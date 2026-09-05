import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hudOutranksSpeechClaim, isSpeechCue, SPEECH_CLAIM_TTL_MS } from './ambient-claim'
import { createEventDeduper } from './event-dedupe'

const quiet = { fromHud: false, hudOpen: false, senderDisplacedByHud: false }

test('only the read-aloud cue is a speech cue', () => {
  assert.equal(isSpeechCue('speak:m1'), true)
  assert.equal(isSpeechCue('sound:turnDone:s1'), false)
})

test('the HUD outranks the app window it hid on its way up', () => {
  assert.equal(
    hudOutranksSpeechClaim('speak:m1', { ...quiet, hudOpen: true, senderDisplacedByHud: true }),
    true
  )
})

test('the HUD never outranks itself', () => {
  assert.equal(
    hudOutranksSpeechClaim('speak:m1', {
      fromHud: true,
      hudOpen: true,
      senderDisplacedByHud: true
    }),
    false
  )
})

test('a peer window the HUD did not displace still speaks', () => {
  assert.equal(hudOutranksSpeechClaim('speak:m1', { ...quiet, hudOpen: true }), false)
})

test('with no HUD up nothing is outranked', () => {
  assert.equal(hudOutranksSpeechClaim('speak:m1', { ...quiet, senderDisplacedByHud: true }), false)
})

test('the turn-end sound is left to the plain first-caller-wins collapse', () => {
  assert.equal(
    hudOutranksSpeechClaim('sound:turnDone:s1', {
      ...quiet,
      hudOpen: true,
      senderDisplacedByHud: true
    }),
    false
  )
})

test('a speech claim survives a throttled window waking well past one second', () => {
  const claimed = createEventDeduper(SPEECH_CLAIM_TTL_MS)

  assert.equal(claimed('speak:m1', 0), false, 'HUD claims and speaks')
  assert.equal(claimed('speak:m1', 8_000), true, 'the late hidden window stays quiet')
})

test('a later reply is never suppressed — the key carries the message id', () => {
  const claimed = createEventDeduper(SPEECH_CLAIM_TTL_MS)

  assert.equal(claimed('speak:m1', 0), false)
  assert.equal(claimed('speak:m2', 8_000), false, 'the next reply speaks')
})
