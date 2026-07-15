import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fontationsWorkaround, MACOS_TAHOE_DARWIN_MAJOR } from './emoji-font-workaround'

test('Tahoe (Darwin 25) disables FontationsFontBackend', () => {
  const result = fontationsWorkaround(MACOS_TAHOE_DARWIN_MAJOR)
  assert.ok(result, 'workaround must apply on Tahoe')
  assert.equal(result.switch, 'disable-features')
  assert.equal(result.value, 'FontationsFontBackend')
})

test('post-Tahoe (Darwin 26+) still disables FontationsFontBackend', () => {
  const result = fontationsWorkaround(MACOS_TAHOE_DARWIN_MAJOR + 1)
  assert.ok(result, 'workaround must apply on future macOS too')
  assert.equal(result.value, 'FontationsFontBackend')
})

test('pre-Tahoe (Darwin 24) does not disable FontationsFontBackend', () => {
  assert.equal(fontationsWorkaround(MACOS_TAHOE_DARWIN_MAJOR - 1), null)
})

test('non-macOS (darwinMajor 0) does not disable FontationsFontBackend', () => {
  assert.equal(fontationsWorkaround(0), null)
})

test('MACOS_TAHOE_DARWIN_MAJOR is 25', () => {
  assert.equal(MACOS_TAHOE_DARWIN_MAJOR, 25)
})
