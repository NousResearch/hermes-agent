import { describe, expect, it } from 'vitest'

import {
  clampPoolLimits,
  parsePoolLimits,
  POOL_LIMITS_BOUNDS,
  POOL_LIMITS_DEFAULTS,
  POOL_LIMITS_MIN
} from './pool-limits'

describe('parsePoolLimits', () => {
  it('falls back to defaults for null/empty/corrupt input', () => {
    expect(parsePoolLimits(null)).toEqual(POOL_LIMITS_DEFAULTS)
    expect(parsePoolLimits(undefined)).toEqual(POOL_LIMITS_DEFAULTS)
    expect(parsePoolLimits('')).toEqual(POOL_LIMITS_DEFAULTS)
    expect(parsePoolLimits('not json {')).toEqual(POOL_LIMITS_DEFAULTS)
  })

  it('parses a valid persisted blob', () => {
    expect(parsePoolLimits(JSON.stringify({ maxBackends: 11, idleMs: 7_200_000 }))).toEqual({
      maxBackends: 11,
      idleMs: 7_200_000
    })
  })

  it('fills missing keys from defaults', () => {
    expect(parsePoolLimits(JSON.stringify({ maxBackends: 5 }))).toEqual({ ...POOL_LIMITS_DEFAULTS, maxBackends: 5 })
    expect(parsePoolLimits('{}')).toEqual(POOL_LIMITS_DEFAULTS)
  })

  it('ignores non-numeric junk instead of NaN-poisoning the pool', () => {
    expect(parsePoolLimits(JSON.stringify({ maxBackends: 'lots', idleMs: null }))).toEqual(POOL_LIMITS_DEFAULTS)
  })
})

describe('clampPoolLimits', () => {
  it('clamps below the floors', () => {
    expect(clampPoolLimits({ maxBackends: 0 }).maxBackends).toBe(POOL_LIMITS_MIN.maxBackends)
    expect(clampPoolLimits({ idleMs: 100 }).idleMs).toBe(POOL_LIMITS_MIN.idleMs)
  })

  it('clamps absurdly high backend counts', () => {
    expect(clampPoolLimits({ maxBackends: 10_000 }).maxBackends).toBeLessThanOrEqual(64)
  })

  it('clamps idleMs to the shared ceiling (7 days)', () => {
    expect(clampPoolLimits({ idleMs: 999_000_000 }).idleMs).toBe(POOL_LIMITS_BOUNDS.idleMsMax)
  })

  it('floors fractional values', () => {
    expect(clampPoolLimits({ maxBackends: 2.9 }).maxBackends).toBe(2)
  })
})
