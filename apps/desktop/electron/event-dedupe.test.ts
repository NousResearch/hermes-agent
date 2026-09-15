import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createAmbientClaimArbiter, createEventDeduper } from './event-dedupe'

test('collapses the same key inside the window (two windows, one event)', () => {
  const isDup = createEventDeduper(1000)

  assert.equal(isDup('input:s1', 0), false, 'first window claims')
  assert.equal(isDup('input:s1', 5), true, 'second window is deduped')
})

test('distinct keys are independent', () => {
  const isDup = createEventDeduper(1000)

  assert.equal(isDup('input:s1', 0), false)
  assert.equal(isDup('approval:s1', 0), false, 'different kind')
  assert.equal(isDup('input:s2', 0), false, 'different session')
})

test('re-fires once the window elapses', () => {
  const isDup = createEventDeduper(1000)

  assert.equal(isDup('turnDone:s1', 0), false)
  assert.equal(isDup('turnDone:s1', 999), true, 'still within window')
  assert.equal(isDup('turnDone:s1', 1000), false, 'window elapsed → fires again')
})

test('clock rollback starts a fresh window instead of suppressing events until catch-up', () => {
  const isDup = createEventDeduper(1000)

  assert.equal(isDup('turnDone:s1', 10_000), false)
  assert.equal(isDup('turnDone:s1', 5_000), false, 'older wall clock cannot make a claim live for 5 seconds')
  assert.equal(isDup('turnDone:s1', 5_100), true, 'dedupe resumes from the rollback observation')
})

test('prunes stale keys so the map cannot grow unbounded', () => {
  const isDup = createEventDeduper(1000)

  for (let i = 0; i < 100; i += 1) {
    // Each far-apart key is pruned before the next, so none linger as duplicates.
    assert.equal(isDup(`turnDone:s${i}`, i * 2000), false)
  }
})

// #99717: the hidden app window under an open HUD claims the same reply late.
test('a spoken reply stays claimed long after the tick window, a beep does not', () => {
  const owns = createAmbientClaimArbiter(1000)

  assert.equal(owns('speak:m1', 0), true, 'HUD renderer claims the reply')
  assert.equal(owns('speak:m1', 5_000), false, 'app window claiming 5 s later stays quiet')
  assert.equal(owns('sound:turnDone:s1', 0), true)
  assert.equal(owns('sound:turnDone:s1', 5_000), true, 'beep keys still re-fire after the tick window')
})

test('a spoken reply can be claimed again once the speech TTL elapses', () => {
  const owns = createAmbientClaimArbiter(1000, 60_000)

  assert.equal(owns('speak:m1', 0), true)
  assert.equal(owns('speak:m1', 60_000), true)
})
