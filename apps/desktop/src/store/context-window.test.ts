import { describe, expect, it } from 'vitest'

import {
  $contextWindowOpen,
  type ContextWindowInfo,
  effectiveContextLength,
  hasContextOverride,
  parseContextLengthInput,
  setContextWindowOpen
} from './context-window'

const info = (over: Partial<ContextWindowInfo> = {}): ContextWindowInfo => ({
  autoContextLength: 200_000,
  configContextLength: 0,
  effectiveContextLength: 200_000,
  model: 'hermes-4',
  provider: 'nous',
  ...over
})

describe('context window override semantics', () => {
  it('treats 0 as auto-detect and reports the resolved auto value as effective', () => {
    const auto = info({ configContextLength: 0, effectiveContextLength: 200_000 })

    expect(hasContextOverride(auto)).toBe(false)
    expect(effectiveContextLength(auto)).toBe(200_000)
  })

  it('reports an explicit override as the effective window', () => {
    const pinned = info({ configContextLength: 64_000, effectiveContextLength: 64_000 })

    expect(hasContextOverride(pinned)).toBe(true)
    expect(effectiveContextLength(pinned)).toBe(64_000)
  })

  it('prefers the backend-resolved effective value over any local derivation', () => {
    // The backend is authoritative: a provider-enforced cap (e.g. Codex OAuth
    // at 272k for a slug models.dev reports as 1.05M) must win over the raw
    // auto figure. Recomputing this client-side is exactly the sibling-path
    // divergence the CLI regression test guards against.
    const capped = info({ autoContextLength: 1_050_000, configContextLength: 0, effectiveContextLength: 272_000 })

    expect(effectiveContextLength(capped)).toBe(272_000)
  })

  it('falls back to the auto value when an older backend omits effective', () => {
    const legacy = info({ autoContextLength: 128_000, configContextLength: 0, effectiveContextLength: 0 })

    expect(effectiveContextLength(legacy)).toBe(128_000)
  })

  it('falls back to the override when an older backend omits effective', () => {
    const legacy = info({ autoContextLength: 128_000, configContextLength: 32_000, effectiveContextLength: 0 })

    expect(effectiveContextLength(legacy)).toBe(32_000)
  })
})

describe('parseContextLengthInput', () => {
  it('maps blank input to 0 (return to auto-detect)', () => {
    expect(parseContextLengthInput('')).toBe(0)
    expect(parseContextLengthInput('   ')).toBe(0)
    expect(parseContextLengthInput('0')).toBe(0)
  })

  it('accepts plain and grouped digits', () => {
    expect(parseContextLengthInput('64000')).toBe(64_000)
    expect(parseContextLengthInput('200,000')).toBe(200_000)
    expect(parseContextLengthInput('200 000')).toBe(200_000)
    expect(parseContextLengthInput(' 128000 ')).toBe(128_000)
  })

  it('rejects values that would persist a garbage pin', () => {
    expect(parseContextLengthInput('abc')).toBeNull()
    expect(parseContextLengthInput('-5')).toBeNull()
    expect(parseContextLengthInput('12.5')).toBeNull()
    expect(parseContextLengthInput('64k')).toBeNull()
  })
})

describe('$contextWindowOpen', () => {
  it('opens and closes the overlay', () => {
    setContextWindowOpen(true)
    expect($contextWindowOpen.get()).toBe(true)

    setContextWindowOpen(false)
    expect($contextWindowOpen.get()).toBe(false)
  })
})
