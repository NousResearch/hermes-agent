import { describe, expect, it } from 'vitest'

import { parseModelLoadWait, providerWaitText } from './provider-wait'

// The load-notice string is minted by the backend
// (agent/chat_completion_helpers._managed_local_load_notice) and parsed
// here — these tests pin the desktop side of that cross-language contract
// (the backend pins its side in tests/hermes_cli/test_load_progress.py).
describe('providerWaitText', () => {
  it('accepts the managed-local load frame', () => {
    const frame = '⏳ loading Qwen3.6-35B-A3B-UD-Q4_K_M into memory — 42% (responses start once the model is loaded)'

    expect(providerWaitText(frame)).toBe(frame)
  })

  it('still accepts classic wait frames and rejects spinner noise', () => {
    expect(providerWaitText('⏳ waiting on local-model — 30s with no output yet')).not.toBe('')
    expect(providerWaitText('◉_◉ cogitating...')).toBe('')
  })
})

describe('parseModelLoadWait', () => {
  it('extracts model and percent from a load frame', () => {
    expect(
      parseModelLoadWait('⏳ loading Qwen3.6-35B-A3B-UD-Q4_K_M into memory — 42% (responses start once the model is loaded)')
    ).toEqual({ kind: 'load', model: 'Qwen3.6-35B-A3B-UD-Q4_K_M', percent: 42 })
  })

  it('extracts the percent from a prefill frame', () => {
    expect(parseModelLoadWait('⚙ processing prompt — 31%')).toEqual({
      kind: 'prefill',
      model: '',
      percent: 31
    })
  })

  it('parses a percentless prefill frame with a null percent (no fake bar)', () => {
    expect(parseModelLoadWait('⚙ processing prompt')).toEqual({
      kind: 'prefill',
      model: '',
      percent: null
    })
  })

  it('returns null for every other wait frame', () => {
    expect(parseModelLoadWait('⏳ waiting on qwen — 30s with no output yet')).toBeNull()
    expect(parseModelLoadWait('⚠ no output from provider for 900s — reconnecting...')).toBeNull()
    expect(parseModelLoadWait('')).toBeNull()
  })

  it('clamps out-of-range percents', () => {
    expect(parseModelLoadWait('⏳ loading m into memory — 999%')?.percent).toBe(100)
  })
})

describe('providerWaitText accepts prefill frames', () => {
  it('passes the ⚙ processing-prompt frame through', () => {
    const frame = '⚙ processing prompt — 31%'

    expect(providerWaitText(frame)).toBe(frame)
  })
})
