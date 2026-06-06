import { describe, expect, it } from 'vitest'

import { resolveVersionBar } from './version-bar'

const copy = {
  clientLabel: (v: string) => `client v${v}`,
  backendLabel: (v: string) => `backend v${v}`,
  unknown: 'unknown'
}

describe('resolveVersionBar', () => {
  it('local mode: shows only the client version (unchanged behaviour)', () => {
    const result = resolveVersionBar({
      appVersion: '0.16.0',
      backendVersion: '0.16.0',
      mode: 'local',
      copy
    })

    expect(result.label).toBe('v0.16.0')
    expect(result.showsBackend).toBe(false)
  })

  it('remote mode: shows BOTH client and backend versions', () => {
    const result = resolveVersionBar({
      appVersion: '0.15.1',
      backendVersion: '0.16.0',
      mode: 'remote',
      copy
    })

    expect(result.label).toContain('client v0.15.1')
    expect(result.label).toContain('backend v0.16.0')
    expect(result.showsBackend).toBe(true)
    expect(result.backendVersion).toBe('0.16.0')
  })

  it('remote mode without a backend version yet: falls back to client-only, no crash', () => {
    const result = resolveVersionBar({
      appVersion: '0.15.1',
      backendVersion: undefined,
      mode: 'remote',
      copy
    })

    expect(result.label).toBe('v0.15.1')
    expect(result.showsBackend).toBe(false)
  })

  it('remote mode where both versions match: still shows both (no implicit hiding)', () => {
    const result = resolveVersionBar({
      appVersion: '0.16.0',
      backendVersion: '0.16.0',
      mode: 'remote',
      copy
    })

    expect(result.label).toContain('client v0.16.0')
    expect(result.label).toContain('backend v0.16.0')
    expect(result.showsBackend).toBe(true)
  })

  it('no client version at all: uses sha fallback for the base, local mode', () => {
    const result = resolveVersionBar({
      appVersion: undefined,
      sha: 'abc1234',
      backendVersion: undefined,
      mode: 'local',
      copy
    })

    expect(result.label).toBe('abc1234')
    expect(result.showsBackend).toBe(false)
  })

  it('exposes a skew flag when remote versions differ', () => {
    const skewed = resolveVersionBar({
      appVersion: '0.15.1',
      backendVersion: '0.16.0',
      mode: 'remote',
      copy
    })
    const aligned = resolveVersionBar({
      appVersion: '0.16.0',
      backendVersion: '0.16.0',
      mode: 'remote',
      copy
    })

    expect(skewed.skew).toBe(true)
    expect(aligned.skew).toBe(false)
  })
})
