import { describe, expect, it } from 'vitest'

import {
  estimateCapabilityBoundOperationalMs,
  estimateOperationalMs,
  makeWaves,
  MAX_EFFECTIVE_WAVE_SIZE,
  MAX_SWEEP_PROBES,
  promotionSweepFeasibility,
  requiredApprovalCount
} from './managed-rollout-waves'

describe('managed rollout wave helpers', () => {
  it('preserves explicit canary and later order', () => {
    expect(makeWaves(['a', 'b', 'c', 'd'], ['b'], 2)).toEqual([['b'], ['a', 'c'], ['d']])
  })

  it('rejects empty and duplicate selections', () => {
    expect(() => makeWaves([], [], 1)).toThrow('invalid-selection')
    expect(() => makeWaves(['a', 'a'], ['a'], 1)).toThrow('invalid-selection')
    expect(() => makeWaves(['a', 'b'], ['a', 'a'], 1)).toThrow('invalid-canaries')
  })

  it('keeps a single target as one wave without inventing a canary', () => {
    expect(makeWaves(['only'], [], 1)).toEqual([['only']])
    expect(makeWaves(['only'], ['only'], 1)).toEqual([['only']])
    expect(() => makeWaves(['only'], ['other'], 1)).toThrow('invalid-canaries')
  })

  it('accepts 120 canaries plus 120 successors and refuses 121 with a named reason', () => {
    const selected = Array.from({ length: MAX_SWEEP_PROBES }, (_, index) => `install-${index}`)
    const canaries = selected.slice(0, MAX_EFFECTIVE_WAVE_SIZE)
    const waves = makeWaves(selected, canaries, MAX_EFFECTIVE_WAVE_SIZE)

    expect(waves[0]).toHaveLength(MAX_EFFECTIVE_WAVE_SIZE)
    expect(waves[1]).toHaveLength(MAX_EFFECTIVE_WAVE_SIZE)
    expect(promotionSweepFeasibility(120, 120)).toMatchObject({ ok: true, probeBudget: 240, reason: null })
    expect(promotionSweepFeasibility(121, 120)).toMatchObject({
      ok: false,
      reason: 'wave-size-exceeds-sweep-budget'
    })
    expect(() => makeWaves(selected, selected.slice(0, 121), 120)).toThrow('wave-size-exceeds-sweep-budget')
    expect(() => makeWaves(selected, ['install-0'], 121)).toThrow('wave-size-exceeds-sweep-budget')
  })

  it('keeps the 500 target selection ceiling separate from the effective wave bound', () => {
    const selected = Array.from({ length: 500 }, (_, index) => `install-${index}`)
    const waves = makeWaves(selected, ['install-0'], MAX_EFFECTIVE_WAVE_SIZE)

    expect(waves.flat()).toEqual(selected)
    expect(Math.max(...waves.map(wave => wave.length))).toBeLessThanOrEqual(MAX_EFFECTIVE_WAVE_SIZE)
  })

  it('counts the mandatory canary approval and later manual boundaries', () => {
    const waves = makeWaves(['a', 'b', 'c'], ['a'], 1)

    expect(requiredApprovalCount(waves, 'manual')).toBe(2)
    expect(requiredApprovalCount(waves, 'auto-if-healthy')).toBe(1)
    expect(requiredApprovalCount([['a']], 'manual')).toBe(0)
  })

  it('does not invent timing for an unmeasured host', () => {
    expect(estimateOperationalMs([['a'], ['b']], { a: 10 }, 1)).toBeNull()
  })

  it('estimates operational work without approval waiting', () => {
    expect(estimateOperationalMs([['a'], ['b', 'c']], { a: 10, b: 20, c: 30 }, 1)).toBe(60)
    expect(estimateOperationalMs([['a', 'b', 'c']], { a: 10, b: 20, c: 30 }, 2)).toBe(40)
  })

  it('rejects an estimate whose concurrency exceeds the current capability', () => {
    expect(
      estimateCapabilityBoundOperationalMs([['a', 'b']], { a: 10, b: 20 }, 4, {
        protocol: 1,
        available: true,
        reason: null,
        maxConcurrency: 1,
        maxInstallations: 10
      })
    ).toBeNull()

    expect(
      estimateCapabilityBoundOperationalMs([['a', 'b']], { a: 10, b: 20 }, 1, {
        protocol: 1,
        available: true,
        reason: null,
        maxConcurrency: 1,
        maxInstallations: 10
      })
    ).toBe(30)
  })
})
