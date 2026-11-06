import { describe, expect, it } from 'vitest'

import { decodePlayerAgentOperation } from './player-operation-decoder'
const contextToken = 'x'.repeat(24)
describe('closed player operation boundary', () => {
  it('translates the captured target and transcript range without routing authority', () => {
    expect(
      decodePlayerAgentOperation({
        type: 'transcript.start',
        context_token: contextToken,
        language: 'en',
        range: { start_seconds: 0, end_seconds: 60 }
      })
    ).toEqual({ type: 'transcript.start', contextToken, language: 'en', range: { startSeconds: 0, endSeconds: 60 } })
    expect(
      decodePlayerAgentOperation({
        type: 'control',
        context_token: contextToken,
        navigation_generation: 3,
        action: 'seek',
        position_seconds: 12
      })
    ).toEqual({ type: 'control', contextToken, navigationGeneration: 3, action: 'seek', positionSeconds: 12 })
  })
  it.each([
    { type: '__proto__' },
    { type: 'context', session_id: 'another-session' },
    { type: 'download.start', context_token: contextToken, url: 'https://example.invalid' },
    { type: 'download.start', context_token: contextToken, quality: { toString: () => 'best' } },
    { type: 'control', context_token: contextToken, navigation_generation: 0, action: { toString: () => 'play' } },
    { type: 'control', context_token: contextToken, navigation_generation: 0, action: 'seek', position_seconds: NaN },
    { type: 'transcript.start', context_token: contextToken, range: { start_seconds: 60, end_seconds: 0 } }
  ])('rejects malformed or authority-bearing inputs %#', input => {
    expect(decodePlayerAgentOperation(input)).toBeNull()
  })
})
