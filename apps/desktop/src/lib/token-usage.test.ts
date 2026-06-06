import { describe, expect, it } from 'vitest'

import { mergeTokenUsagePayload, usageFromTokenUsagePayload } from './token-usage'

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
})
