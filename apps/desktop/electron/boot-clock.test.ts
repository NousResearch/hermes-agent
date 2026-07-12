/**
 * Tests for electron/boot-clock.ts.
 *
 * Run with: node --test electron/boot-clock.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Pure boot-milestone clock + cache-observability line formatters. A fake
 * `now()` makes offset assertions exact — no wall-clock flake.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { createBootClock, formatCacheHit, formatCacheDivergence } from './boot-clock'

test('mark() formats [boot:t+<ms>ms] <milestone> from a fake clock', () => {
  let t = 1000
  const clock = createBootClock({ now: () => t, t0: 1000 })
  assert.equal(clock.mark('process-start'), '[boot:t+0ms] process-start')
  t = 2412
  assert.equal(clock.mark('window-created'), '[boot:t+1412ms] window-created')
})

test('mark() appends a trimmed detail when provided', () => {
  let t = 0
  const clock = createBootClock({ now: () => t, t0: 0 })
  t = 6000
  assert.equal(clock.mark('list-loaded', '  48/2576  '), '[boot:t+6000ms] list-loaded 48/2576')
})

test('mark() omits an empty/nullish detail', () => {
  const clock = createBootClock({ now: () => 0, t0: 0 })
  assert.equal(clock.mark('ready', ''), '[boot:t+0ms] ready')
  assert.equal(clock.mark('ready', undefined), '[boot:t+0ms] ready')
})

test('elapsedMs() rounds and never goes negative even if the clock steps back', () => {
  let t = 500
  const clock = createBootClock({ now: () => t, t0: 500 })
  t = 500.4
  assert.equal(clock.elapsedMs(), 0)
  t = 500.6
  assert.equal(clock.elapsedMs(), 1)
  // A swapped/backwards clock must not emit a negative offset.
  t = 100
  assert.equal(clock.elapsedMs(), 0)
})

test('t0 defaults to now() when not supplied', () => {
  let t = 42
  const clock = createBootClock({ now: () => t })
  assert.equal(clock.t0, 42)
  t = 142
  assert.equal(clock.elapsedMs(), 100)
})

test('formatCacheHit: miss and hit-with-rows', () => {
  assert.equal(formatCacheHit(false), '[boot] cache-miss')
  assert.equal(formatCacheHit(true, 48), '[boot] cache-hit rows=48')
  // rows defaults to 0 and clamps garbage.
  assert.equal(formatCacheHit(true), '[boot] cache-hit rows=0')
  assert.equal(formatCacheHit(true, -5), '[boot] cache-hit rows=0')
  assert.equal(formatCacheHit(true, 12.7), '[boot] cache-hit rows=13')
})

test('formatCacheDivergence: 0 means the cache matched live exactly', () => {
  assert.equal(formatCacheDivergence(0), '[boot] cache-divergence rows=0')
  assert.equal(formatCacheDivergence(3), '[boot] cache-divergence rows=3')
  assert.equal(formatCacheDivergence(-1), '[boot] cache-divergence rows=0')
  assert.equal(formatCacheDivergence(2.4), '[boot] cache-divergence rows=2')
})
