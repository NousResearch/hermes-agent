import { describe, expect, it } from 'vitest'

import { isLargeChat, LARGE_CHAT_TOKENS } from './session-token-size'

describe('isLargeChat', () => {
  it('flags a chat at or above a full context window', () => {
    expect(isLargeChat(LARGE_CHAT_TOKENS)).toBe(true)
    expect(isLargeChat(LARGE_CHAT_TOKENS + 1)).toBe(true)
    expect(isLargeChat(500_000)).toBe(true)
  })

  it('leaves a normal working chat alone', () => {
    // The two sizes the feature was drawn from — busy, but the model is still
    // comfortable — must NOT nag.
    expect(isLargeChat(97_600)).toBe(false)
    expect(isLargeChat(133_300)).toBe(false)
    expect(isLargeChat(LARGE_CHAT_TOKENS - 1)).toBe(false)
  })

  it('degrades to "not large" for missing/empty/negative counts', () => {
    expect(isLargeChat(0)).toBe(false)
    expect(isLargeChat(undefined)).toBe(false)
    expect(isLargeChat(null)).toBe(false)
  })
})
