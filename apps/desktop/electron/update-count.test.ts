import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveBehindCount, resolveUpdateCurrentSha, shouldCountCommits } from './update-count'

test('packaged builds compare the installed build stamp instead of a stale checkout', () => {
  const installedSha = 'a'.repeat(40)

  assert.equal(
    resolveUpdateCurrentSha({
      checkoutSha: 'b'.repeat(40),
      installStampSha: installedSha,
      isPackaged: true
    }),
    installedSha
  )
})

test('development builds compare the checkout being run', () => {
  assert.equal(
    resolveUpdateCurrentSha({
      checkoutSha: 'development-head',
      installStampSha: 'old-build-stamp',
      isPackaged: false
    }),
    'development-head'
  )
})

test('packaged builds without a valid stamp fall back to checkout HEAD', () => {
  assert.equal(
    resolveUpdateCurrentSha({
      checkoutSha: 'checkout-head',
      installStampSha: '',
      isPackaged: true
    }),
    'checkout-head'
  )
})

test('packaged builds reject the all-zero non-Git build sentinel', () => {
  assert.equal(
    resolveUpdateCurrentSha({
      checkoutSha: 'checkout-head',
      installStampSha: '0'.repeat(40),
      isPackaged: true
    }),
    'checkout-head'
  )
})

test('packaged builds reject malformed install stamp object IDs', () => {
  for (const installStampSha of ['installed-build', 'abc1234', 'a'.repeat(39), 'a'.repeat(41), `${'a'.repeat(39)}g`]) {
    assert.equal(
      resolveUpdateCurrentSha({ checkoutSha: 'checkout-head', installStampSha, isPackaged: true }),
      'checkout-head'
    )
  }
})

test('packaged builds accept SHA-256 repository object IDs', () => {
  const installedSha = 'c'.repeat(64)

  assert.equal(
    resolveUpdateCurrentSha({
      checkoutSha: 'checkout-head',
      installStampSha: installedSha,
      isPackaged: true
    }),
    installedSha
  )
})

// FAIL-BEFORE: pre-fix the function did `Number.parseInt(countStr) || 0`
// unconditionally, so a shallow checkout with no merge-base surfaced the bogus
// rev-list count (e.g. 12104). This asserts the new shallow/no-merge-base branch.
test('shallow checkout with no merge-base does NOT trust the bogus rev-list count', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'aaa',
      targetSha: 'bbb',
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
      hasMergeBase: true
    }),
    3
  )
})

// A full checkout can also have unrelated history (for example, a custom
// source checkout next to a packaged app built from the official repository).
// `HEAD..origin/main` then enumerates the remote's whole ancestry just like the
// shallow-clone failure mode, so the exact count is not meaningful.
test('full checkout without a merge-base rejects the whole-ancestry count', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '14829',
      currentSha: 'custom-root',
      targetSha: 'official-tip',
      hasMergeBase: false
    }),
    1
  )
})

test('full (non-shallow) clone keeps the exact count path unchanged', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '7',
      currentSha: 'aaa',
      targetSha: 'bbb',
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
  assert.equal(shouldCountCommits({ hasMergeBase: false }), false)
})

test('shallow checkout WITH a merge-base still runs the count', () => {
  assert.equal(shouldCountCommits({ hasMergeBase: true }), true)
})

test('full checkout with a merge-base runs the exact count', () => {
  assert.equal(shouldCountCommits({ hasMergeBase: true }), true)
})

test('full checkout without a merge-base skips the meaningless count', () => {
  assert.equal(shouldCountCommits({ hasMergeBase: false }), false)
})

// The skip path produces an empty countStr; resolveBehindCount must NOT trust
// it and must fall through to the SHA compare (mirrors the live call site).
test('skipped-count path resolves via SHA compare, never via empty countStr', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'aaa',
      targetSha: 'bbb',
      hasMergeBase: false
    }),
    1
  )
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'same',
      targetSha: 'same',
      hasMergeBase: false
    }),
    0
  )
})
