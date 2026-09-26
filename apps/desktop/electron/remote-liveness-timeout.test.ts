import { describe, expect, it } from 'vitest'

import {
  clampRemoteLivenessTimeoutMs,
  parsePersistedRemoteLivenessTimeoutMs,
  REMOTE_LIVENESS_TIMEOUT_BOUNDS,
  REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS,
  resolveRemoteLivenessTimeoutMsFromEnv
} from './remote-liveness-timeout'

describe('clampRemoteLivenessTimeoutMs', () => {
  it('falls back to the default for non-numeric or non-positive input', () => {
    expect(clampRemoteLivenessTimeoutMs('nope')).toBe(REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS)
    expect(clampRemoteLivenessTimeoutMs(0)).toBe(REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS)
    expect(clampRemoteLivenessTimeoutMs(-5)).toBe(REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS)
    expect(clampRemoteLivenessTimeoutMs(Number.NaN)).toBe(REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS)
  })

  it('clamps below the floor', () => {
    expect(clampRemoteLivenessTimeoutMs(1)).toBe(REMOTE_LIVENESS_TIMEOUT_BOUNDS.min)
  })

  it('clamps runaway values so a typo cannot hang dispatch forever', () => {
    expect(clampRemoteLivenessTimeoutMs(999_999)).toBe(REMOTE_LIVENESS_TIMEOUT_BOUNDS.max)
  })

  it('floors fractional values', () => {
    expect(clampRemoteLivenessTimeoutMs(30_000.9)).toBe(30_000)
  })

  it('passes through a value already within bounds', () => {
    expect(clampRemoteLivenessTimeoutMs(30_000)).toBe(30_000)
  })
})

describe('resolveRemoteLivenessTimeoutMsFromEnv', () => {
  it('defaults to 10s when unset', () => {
    expect(resolveRemoteLivenessTimeoutMsFromEnv({})).toBe(10_000)
  })

  it('honours a configured HERMES_REMOTE_LIVENESS_TIMEOUT_MS', () => {
    expect(resolveRemoteLivenessTimeoutMsFromEnv({ HERMES_REMOTE_LIVENESS_TIMEOUT_MS: '30000' })).toBe(30_000)
  })

  it('falls back to the default on missing, zero, or unparsable values', () => {
    expect(resolveRemoteLivenessTimeoutMsFromEnv({ HERMES_REMOTE_LIVENESS_TIMEOUT_MS: '0' })).toBe(10_000)
    expect(resolveRemoteLivenessTimeoutMsFromEnv({ HERMES_REMOTE_LIVENESS_TIMEOUT_MS: 'nope' })).toBe(10_000)
    expect(resolveRemoteLivenessTimeoutMsFromEnv({ HERMES_REMOTE_LIVENESS_TIMEOUT_MS: '' })).toBe(10_000)
  })

  it('clamps runaway values from the env var too', () => {
    expect(resolveRemoteLivenessTimeoutMsFromEnv({ HERMES_REMOTE_LIVENESS_TIMEOUT_MS: '999999' })).toBe(120_000)
  })
})

describe('parsePersistedRemoteLivenessTimeoutMs', () => {
  it('returns null for null/empty/corrupt input so the caller falls through to the env fallback', () => {
    expect(parsePersistedRemoteLivenessTimeoutMs(null)).toBeNull()
    expect(parsePersistedRemoteLivenessTimeoutMs(undefined)).toBeNull()
    expect(parsePersistedRemoteLivenessTimeoutMs('')).toBeNull()
    expect(parsePersistedRemoteLivenessTimeoutMs('not json {')).toBeNull()
  })

  it('returns null when the persisted blob has no numeric timeoutMs', () => {
    expect(parsePersistedRemoteLivenessTimeoutMs(JSON.stringify({}))).toBeNull()
    expect(parsePersistedRemoteLivenessTimeoutMs(JSON.stringify({ timeoutMs: 'lots' }))).toBeNull()
  })

  it('parses and clamps a valid persisted blob', () => {
    expect(parsePersistedRemoteLivenessTimeoutMs(JSON.stringify({ timeoutMs: 30_000 }))).toBe(30_000)
    expect(parsePersistedRemoteLivenessTimeoutMs(JSON.stringify({ timeoutMs: 999_999 }))).toBe(120_000)
  })
})
