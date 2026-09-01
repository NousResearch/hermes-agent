import { expect, test } from 'vitest'

import {
  DEFAULT_PROFILE_BACKEND_POOL_MAX,
  normalizeProfileBackendPoolMax,
  readProfileBackendPoolSettings,
  writeProfileBackendPoolSettings
} from './profile-backend-pool-settings'

test('pool max accepts zero as the explicit unlimited value', () => {
  expect(normalizeProfileBackendPoolMax(0)).toBe(0)
  expect(normalizeProfileBackendPoolMax('0')).toBe(0)
  expect(normalizeProfileBackendPoolMax(10)).toBe(10)
})

test('pool max rejects negative, fractional, unsafe, and empty values', () => {
  expect(normalizeProfileBackendPoolMax(-1)).toBe(DEFAULT_PROFILE_BACKEND_POOL_MAX)
  expect(normalizeProfileBackendPoolMax(1.5)).toBe(DEFAULT_PROFILE_BACKEND_POOL_MAX)
  expect(normalizeProfileBackendPoolMax(Number.MAX_SAFE_INTEGER + 1)).toBe(DEFAULT_PROFILE_BACKEND_POOL_MAX)
  expect(normalizeProfileBackendPoolMax('')).toBe(DEFAULT_PROFILE_BACKEND_POOL_MAX)
  expect(normalizeProfileBackendPoolMax('not-a-number')).toBe(DEFAULT_PROFILE_BACKEND_POOL_MAX)
})

test('settings round-trip preserves ten and zero without truthiness fallback', () => {
  const files = new Map<string, string>()
  const write = (filePath: string, content: string) => files.set(filePath, content)
  const rename = (from: string, to: string) => {
    const content = files.get(from)
    if (content === undefined) {
      throw new Error('temporary file missing')
    }
    files.delete(from)
    files.set(to, content)
  }

  writeProfileBackendPoolSettings('/tmp/ares-settings', 10, write, rename, () => undefined)
  expect(readProfileBackendPoolSettings('/tmp/ares-settings', filePath => files.get(filePath)!)).toEqual({
    maxBackends: 10
  })

  writeProfileBackendPoolSettings('/tmp/ares-settings', 0, write, rename, () => undefined)
  expect(readProfileBackendPoolSettings('/tmp/ares-settings', filePath => files.get(filePath)!)).toEqual({
    maxBackends: 0
  })
})

test('malformed settings fall back to the safe default', () => {
  expect(
    readProfileBackendPoolSettings('/tmp/ares-settings', () => '{"max_backends": -2}')
  ).toEqual({ maxBackends: DEFAULT_PROFILE_BACKEND_POOL_MAX })
})
