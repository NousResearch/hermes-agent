import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveBehindCount, shouldCountCommits } from './update-count'

// FAIL-BEFORE: pre-fix `shouldCountCommits` returned `!(isShallow && !hasMergeBase)`,
// so a SHALLOW checkout that happened to share a merge-base with the tip STILL ran
// `rev-list --count` and surfaced the bogus thousands-of-commits number
// (e.g. the real-world 4650 a shallow pin reported). The CLI's banner.py already
// compares tip SHAs for ANY shallow clone — this aligns the desktop with that.
// A shallow checkout must ALWAYS skip the count and fall back to a SHA compare,
// regardless of whether a merge-base exists.

test('shallow checkout with no merge-base does NOT trust the bogus rev-list count', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: false
    }),
    1
  )
})

// The previously-missing case: shallow + merge-base was the live bug (4650).
test('shallow checkout WITH a merge-base does NOT trust the bogus rev-list count', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '4650',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: true
    }),
    1
  )
})

test('shallow checkout behind by a bogus count but identical SHA reports up-to-date', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '4650',
      currentSha: 'abc',
      targetSha: 'abc',
      isShallow: true,
      hasMergeBase: true
    }),
    0
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
// Any shallow checkout (installer / binary / desktop pin) must skip it — the
// local history is truncated, so the count is never meaningful. Only full
// clones run it.
test('shallow checkout SKIPS the rev-list count (no merge-base)', () => {
  assert.equal(shouldCountCommits({ isShallow: true, hasMergeBase: false }), false)
})

test('shallow checkout SKIPS the rev-list count (with merge-base — the live bug)', () => {
  assert.equal(shouldCountCommits({ isShallow: true, hasMergeBase: true }), false)
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
    1
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
