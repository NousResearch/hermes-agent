import assert from 'node:assert/strict'
import { test } from 'vitest'

import { nativeQuad } from '../../../scripts/msix-shared.mjs'

// 2026-09-22T00:14:03Z, worked by hand: day-of-year 264 (2026 is not a leap
// year; Jan 1 is hour 0), so hour-of-year is 264*24 = 6336. The 14th minute
// holds 14*60+3 = 843 seconds.
const EPOCH = Date.parse('2026-09-22T00:14:03Z') / 1000

test('stable keeps the store quad while canary uses yy.mmdd.hh.mmss', () => {
  assert.equal(nativeQuad('v0.21.5', EPOCH), '2026.6336.843.0')
  assert.equal(nativeQuad('v0.21.4+canary.20260922T001403Z', EPOCH), '26.922.0.1403')
})

test('historical release identities are rejected', () => {
  assert.throws(() => nativeQuad('v0.21.4-canary.20260922T001403Z', EPOCH))
  assert.throws(() => nativeQuad('v2026.9.21', EPOCH))
  assert.throws(() => nativeQuad('v2026.9.21+canary.20260922T001403Z', EPOCH))
})

test('a later canary sorts above an earlier canary across a month boundary', () => {
  const earlier = nativeQuad(
    'v0.21.4+canary.20260131T235959Z',
    Date.parse('2026-01-31T23:59:59Z') / 1000
  ).split('.').map(Number)
  const later = nativeQuad(
    'v0.21.4+canary.20260201T000000Z',
    Date.parse('2026-02-01T00:00:00Z') / 1000
  ).split('.').map(Number)
  const first = later.findIndex((value, index) => value !== earlier[index])
  assert.ok(first >= 0)
  assert.ok(later[first] > earlier[first])
})

test('every field stays inside 16 bits', () => {
  for (const stamp of ['2026-01-01T00:00:00Z', '2026-12-31T23:59:59Z', '2028-12-31T23:59:59Z']) {
    for (const ref of ['v0.21.5', `v0.21.4+canary.${stamp.replace(/[-:]/g, '').replace('.000', '')}`]) {
      const parts = nativeQuad(ref, Date.parse(stamp) / 1000).split('.').map(Number)
      assert.equal(parts.length, 4)
      assert.ok(parts.every(value => value >= 0 && value <= 65535))
    }
  }
})
