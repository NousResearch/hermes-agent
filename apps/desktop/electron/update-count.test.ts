import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveBehindCount, shouldCountCommits } from './update-count'

// FAIL-BEFORE: the shallow/no-merge-base branch returned the sentinel `1`,
// which the UI rendered as a literal "1 change included" even when the true
// count was far higher (e.g. 90). An update IS available here, but its exact
// size is unknown — the only honest value is `null`.
test('shallow checkout with no merge-base reports null (unknown count), not a fake 1', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: false
    }),
    null
  )
})

test('shallow checkout with no merge-base but identical SHA reports up-to-date', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'abc',
      targetSha: 'abc',
      isShallow: true,
      hasMergeBase: false
    }),
    0
  )
})

test('shallow checkout WITH a merge-base keeps the exact count (reliable)', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '3',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: true
    }),
    3
  )
})

test('full (non-shallow) clone keeps the exact count path unchanged', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '7',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: false,
      hasMergeBase: true
    }),
    7
  )
})

test('up-to-date full clone reports 0', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '0',
      currentSha: 'x',
      targetSha: 'x',
      isShallow: false,
      hasMergeBase: true
    }),
    0
  )
})

test('non-numeric count falls back to 0 (defensive, unchanged behaviour)', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: false,
      hasMergeBase: true
    }),
    0
  )
})

// shouldCountCommits gates the expensive `rev-list --count` in checkUpdates().
// FAIL-BEFORE: in the shallow + no-merge-base case the caller ran rev-list
// unconditionally and discarded the bogus result; this predicate lets the
// caller SKIP the whole-ancestry enumeration in exactly that case (#51922).
test('shallow checkout with no merge-base SKIPS the rev-list count', () => {
  assert.equal(shouldCountCommits({ isShallow: true, hasMergeBase: false }), false)
})

test('shallow checkout WITH a merge-base still runs the count', () => {
  assert.equal(shouldCountCommits({ isShallow: true, hasMergeBase: true }), true)
})

test('full (non-shallow) clone always runs the count', () => {
  assert.equal(shouldCountCommits({ isShallow: false, hasMergeBase: true }), true)
  assert.equal(shouldCountCommits({ isShallow: false, hasMergeBase: false }), true)
})

// The skip path produces an empty countStr; resolveBehindCount must NOT trust
// it and must fall through to the SHA compare (mirrors the live call site).
test('skipped-count path resolves via SHA compare, never via empty countStr', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: false
    }),
    null
  )
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'same',
      targetSha: 'same',
      isShallow: true,
      hasMergeBase: false
    }),
    0
  )
})
