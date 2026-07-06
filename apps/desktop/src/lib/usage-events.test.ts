import { describe, expect, it } from 'vitest'

import { usageFromTokenUsagePayload } from './usage-events'

describe('usageFromTokenUsagePayload', () => {
  it('maps gateway token.usage payloads onto desktop usage stats', () => {
    expect(
      usageFromTokenUsagePayload({
        context_length: 131_072,
        context_pct: 9.4,
        context_tokens: 12_345,
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 120
      })
    ).toEqual({
      context_max: 131_072,
      context_percent: 9.4,
      context_used: 12_345,
      input: 100,
      output: 20,
      total: 120
    })
  })

  it('derives context percent when the backend omits it', () => {
    expect(
      usageFromTokenUsagePayload({
        context_length: 200_000,
        context_tokens: 50_000
      })
    ).toMatchObject({
      context_max: 200_000,
      context_percent: 25,
      context_used: 50_000
    })
  })
})
