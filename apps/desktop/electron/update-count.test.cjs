'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { resolveBehindCount } = require('./update-count.cjs')

// FAIL-BEFORE: pre-fix the function did `Number.parseInt(countStr) || 0`
// unconditionally, so a shallow checkout with no merge-base surfaced the bogus
// rev-list count (e.g. 12104). This asserts the new shallow/no-merge-base branch.
test('shallow checkout with no merge-base does NOT trust the bogus rev-list count', () => {
  assert.equal(resolveBehindCount({
    countStr: '12104', currentSha: 'aaa', targetSha: 'bbb',
    isShallow: true, hasMergeBase: false,
  }), 1)
})

test('shallow checkout with no merge-base but identical SHA reports up-to-date', () => {
  assert.equal(resolveBehindCount({
    countStr: '12104', currentSha: 'abc', targetSha: 'abc',
    isShallow: true, hasMergeBase: false,
  }), 0)
})

test('shallow checkout WITH a merge-base keeps the exact count (reliable)', () => {
  assert.equal(resolveBehindCount({
    countStr: '3', currentSha: 'aaa', targetSha: 'bbb',
    isShallow: true, hasMergeBase: true,
  }), 3)
})

test('full (non-shallow) clone keeps the exact count path unchanged', () => {
  assert.equal(resolveBehindCount({
    countStr: '7', currentSha: 'aaa', targetSha: 'bbb',
    isShallow: false, hasMergeBase: true,
  }), 7)
})

test('up-to-date full clone reports 0', () => {
  assert.equal(resolveBehindCount({
    countStr: '0', currentSha: 'x', targetSha: 'x',
    isShallow: false, hasMergeBase: true,
  }), 0)
})

test('non-numeric count falls back to 0 (defensive, unchanged behaviour)', () => {
  assert.equal(resolveBehindCount({
    countStr: '', currentSha: 'aaa', targetSha: 'bbb',
    isShallow: false, hasMergeBase: true,
  }), 0)
})
