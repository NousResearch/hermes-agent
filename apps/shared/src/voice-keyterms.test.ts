import { describe, expect, it } from 'vitest'

import { MAX_KEYTERM_CHARS, realtimeVoiceContextKeyterms } from './voice-keyterms'

describe('realtimeVoiceContextKeyterms', () => {
  it('never emits a term over the xAI per-term cap', () => {
    const long = 'a'.repeat(MAX_KEYTERM_CHARS * 3)

    for (const term of realtimeVoiceContextKeyterms(`/home/user/${long}`)) {
      expect(term.length).toBeLessThanOrEqual(MAX_KEYTERM_CHARS)
    }
  })

  it('extracts distinctive workspace terms and drops generic path segments', () => {
    expect(
      realtimeVoiceContextKeyterms('/home/user/hermes-agent/apps/desktop')
    ).toEqual(['hermes-agent'])
  })

  it('is empty without a workspace', () => {
    expect(realtimeVoiceContextKeyterms(null)).toEqual([])
  })
})
