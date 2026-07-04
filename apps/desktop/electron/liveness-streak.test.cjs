'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const { createLivenessStreak } = require('./liveness-streak.cjs')

test('does not drop until the threshold of consecutive failures is reached', () => {
  const streak = createLivenessStreak({ threshold: 3 })

  assert.deepEqual(streak.recordFailure('http://a'), { drop: false, count: 1, threshold: 3, firstOfStreak: true })
  assert.deepEqual(streak.recordFailure('http://a'), { drop: false, count: 2, threshold: 3, firstOfStreak: false })
  assert.deepEqual(streak.recordFailure('http://a'), { drop: true, count: 3, threshold: 3, firstOfStreak: false })
})

test('a success resets the streak', () => {
  const streak = createLivenessStreak({ threshold: 3 })

  streak.recordFailure('http://a')
  streak.recordFailure('http://a')
  streak.recordSuccess()

  const next = streak.recordFailure('http://a')
  assert.equal(next.drop, false)
  assert.equal(next.count, 1)
  assert.equal(next.firstOfStreak, true)
})

test('a key change restarts the streak (different backend being probed)', () => {
  const streak = createLivenessStreak({ threshold: 2 })

  streak.recordFailure('http://a')
  const next = streak.recordFailure('http://b')
  assert.equal(next.drop, false)
  assert.equal(next.count, 1)
})

test('dropping resets the streak so the rebuilt connection gets a fresh budget', () => {
  const streak = createLivenessStreak({ threshold: 2 })

  streak.recordFailure('http://a')
  assert.equal(streak.recordFailure('http://a').drop, true)

  const next = streak.recordFailure('http://a')
  assert.equal(next.drop, false)
  assert.equal(next.count, 1)
  assert.equal(next.firstOfStreak, true)
})

test('threshold clamps to at least 1 and tolerates junk input', () => {
  for (const threshold of [0, -5, NaN, undefined, 'nope']) {
    const streak = createLivenessStreak({ threshold })
    assert.equal(streak.recordFailure('http://a').drop, true, `threshold ${threshold} should drop on first failure`)
  }
})

test('threshold of 1 restores drop-on-first-failure', () => {
  const streak = createLivenessStreak({ threshold: 1 })
  const result = streak.recordFailure('http://a')
  assert.equal(result.drop, true)
  assert.equal(result.firstOfStreak, true)
})
