import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveBehindCount, shouldCountCommits } from './update-count'

// FAIL-BEFORE: pre-fix the function did `Number.parseInt(countStr) || 0`
// unconditionally, so a shallow checkout with no merge-base surfaced the bogus
// rev-list count (e.g. 12104). This asserts the new shallow/no-merge-base branch.
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

// FAIL-BEFORE: a shallow checkout with local commits on top of the remote tip
// (e.g. cherry-picked fixes) had differing tip SHAs, so the shallow fallback
// reported "update available" even though HEAD already contains the remote tip.
// The caller now passes isAncestor=true from `git merge-base --is-ancestor`,
// and the fallback must treat that as up-to-date (mirrors hermes_cli.gitlock).
test('shallow checkout whose remote tip is an ancestor of HEAD reports up-to-date', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'abc',
      targetSha: 'def',
      isShallow: true,
      hasMergeBase: false,
      isAncestor: true
    }),
    0
  )
})

test('shallow checkout whose remote tip is NOT an ancestor still reports update available', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'abc',
      targetSha: 'def',
      isShallow: true,
      hasMergeBase: false,
      isAncestor: false
    }),
    1
  )
})

test('isAncestor is ignored when a merge-base exists (count path wins)', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '2',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true,
      hasMergeBase: true,
      isAncestor: true
    }),
    2
  )
})

