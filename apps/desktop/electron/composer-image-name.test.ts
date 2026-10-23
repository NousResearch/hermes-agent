/**
 * Tests for electron/composer-image-name.ts.
 *
 * Run with: vitest run --project electron composer-image-name
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Regression (issue #96403): pasted composer screenshots were named with a
 * UTC timestamp (Date.toISOString()), so the filename disagreed with the
 * file's mtime, which the OS shows in local time — e.g. 8 hours off for
 * UTC+8 users. The stamp must be built from the date's LOCAL components.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { composerImageTimestamp } from './composer-image-name'

// A duck-typed Date whose local and UTC readings deliberately differ, so the
// test distinguishes local-time formatting from toISOString() on any host,
// regardless of the machine's own timezone (CI runners often run in UTC,
// where a real Date could not tell the two apart).
const LOCAL_VS_UTC_SKEWED = {
  getFullYear: () => 2026,
  getMonth: () => 7, // August, 0-based
  getDate: () => 18,
  getHours: () => 20,
  getMinutes: () => 52,
  getSeconds: () => 56,
  getMilliseconds: () => 282,
  toISOString: () => '2026-08-18T12:52:56.282Z'
} as unknown as Date

test('composerImageTimestamp renders local wall-clock components, not UTC', () => {
  // Local reading is 20:52:56.282; the UTC reading (what a toISOString()-based
  // stamp would produce) is 12:52:56.282. The filename must carry the local
  // one so it agrees with the file's mtime.
  assert.equal(composerImageTimestamp(LOCAL_VS_UTC_SKEWED), '2026-08-18_20-52-56-282')
})

test('composerImageTimestamp zero-pads every component', () => {
  const singleDigitComponents = {
    getFullYear: () => 2026,
    getMonth: () => 0, // January
    getDate: () => 3,
    getHours: () => 4,
    getMinutes: () => 5,
    getSeconds: () => 6,
    getMilliseconds: () => 7
  } as unknown as Date

  assert.equal(composerImageTimestamp(singleDigitComponents), '2026-01-03_04-05-06-007')
})

test('composerImageTimestamp keeps the legacy filename shape for real dates', () => {
  // Same shape the toISOString()-based stamp produced, so existing tooling
  // that parses composer_* filenames keeps working.
  assert.match(composerImageTimestamp(), /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3}$/)
})
