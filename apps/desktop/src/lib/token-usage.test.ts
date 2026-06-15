import { describe, expect, it } from 'vitest'

import { mergeTokenUsagePayload, mergeUsageSnapshot, usageFromTokenUsagePayload } from './token-usage'

describe('usageFromTokenUsagePayload', () => {
  it('maps gateway token.usage payload fields into statusbar usage stats', () => {
    expect(
      usageFromTokenUsagePayload({
        context_length: 131_072,
        context_pct: 49.9,
        context_tokens: 65_432,
        input_tokens: 1_200,
        output_tokens: 34,
        total_tokens: 1_234
      })
    ).toEqual({
      context_max: 131_072,
      context_percent: 49.9,
      context_used: 65_432,
      input: 1_200,
      output: 34,
      total: 1_234
    })
  })

  it('derives context percent when the backend omits context_pct', () => {
    expect(
      usageFromTokenUsagePayload({
        context_length: 200_000,
        context_tokens: 50_000,
        input_tokens: 50_000,
        total_tokens: 50_000
      })
    ).toMatchObject({
      context_max: 200_000,
      context_percent: 25,
      context_used: 50_000
    })
  })

  it('preserves existing usage when a malformed payload arrives', () => {
    const current = { calls: 2, input: 10, output: 20, total: 30 }

    expect(mergeTokenUsagePayload(current, { input_tokens: Number.NaN })).toBe(current)
  })

  it('keeps live context usage from moving backwards on older snapshots', () => {
    const current = {
      calls: 2,
      context_max: 100_000,
      context_percent: 67,
      context_used: 67_000,
      input: 50_000,
      output: 1_000,
      total: 51_000
    }

    expect(
      mergeUsageSnapshot(current, {
        calls: 3,
        context_max: 100_000,
        context_percent: 48,
        context_used: 48_000,
        input: 60_000,
        output: 2_000,
        total: 62_000
      })
    ).toEqual({
      calls: 3,
      context_max: 100_000,
      context_percent: 67,
      context_used: 67_000,
      input: 60_000,
      output: 2_000,
      total: 62_000
    })
  })

  it('allows context usage to drop when compression advances', () => {
    const current = {
      calls: 2,
      compressions: 0,
      context_max: 100_000,
      context_percent: 67,
      context_used: 67_000,
      input: 50_000,
      output: 1_000,
      total: 51_000
    }

    expect(
      mergeUsageSnapshot(current, {
        compressions: 1,
        context_max: 100_000,
        context_percent: 22,
        context_used: 22_000
      })
    ).toMatchObject({
      compressions: 1,
      context_percent: 22,
      context_used: 22_000
    })
  })

  it('allows explicit replacements to lower context usage during session changes', () => {
    const current = {
      calls: 2,
      context_max: 100_000,
      context_percent: 67,
      context_used: 67_000,
      input: 50_000,
      output: 1_000,
      total: 51_000
    }

    expect(
      mergeUsageSnapshot(
        current,
        {
          calls: 0,
          context_max: 100_000,
          context_percent: 12,
          context_used: 12_000,
          input: 10_000,
          output: 100,
          total: 10_100
        },
        { allowContextDecrease: true }
      )
    ).toMatchObject({
      context_percent: 12,
      context_used: 12_000,
      total: 10_100
    })
  })
})
