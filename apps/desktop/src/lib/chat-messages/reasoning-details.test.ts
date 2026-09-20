import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { toChatMessages } from './hydration'
import { reasoningTextFromDetails } from './parts'

describe('reasoning-details display text', () => {
  it('extracts prose from decoded or serialized envelopes without exposing opaque fields', () => {
    const details = [
      { type: 'reasoning.summary', summary: 'Check the inputs.', signature: 'opaque-token' },
      { type: 'thinking', thinking: '', signature: 'opaque-token-2' },
      { type: 'redacted_thinking', data: 'encrypted-replay-data' },
      { type: 'reasoning.text', text: 'Return the result.' }
    ]

    for (const value of [details, JSON.stringify(details)]) {
      const text = reasoningTextFromDetails(value)

      expect(text).toBe('Check the inputs.\n\nReturn the result.')
      expect(text).not.toContain('opaque-token')
      expect(text).not.toContain('encrypted-replay-data')
    }

    expect(reasoningTextFromDetails('Plain persisted reasoning.')).toBe('Plain persisted reasoning.')
    expect(reasoningTextFromDetails('[{"type":"thinking","thinking":"truncated"')).toBe('')
  })

  it('hydrates only extracted reasoning text and preserves canonical-field precedence', () => {
    const reasoningParts = (message: Partial<SessionMessage>): string[] =>
      toChatMessages([{ role: 'assistant', content: 'Answer.', ...message } as SessionMessage])
        .flatMap(item => item.parts)
        .filter(part => part.type === 'reasoning')
        .map(part => (part as { text: string }).text)

    expect(reasoningParts({ reasoning_details: [{ type: 'thinking', signature: 'opaque-only' }] })).toEqual([])
    expect(
      reasoningParts({ reasoning_details: JSON.stringify([{ type: 'thinking', thinking: 'Visible thought.' }]) })
    ).toEqual(['Visible thought.'])
    expect(
      reasoningParts({
        reasoning: 'Canonical reasoning.',
        reasoning_details: JSON.stringify([{ type: 'thinking', thinking: 'Fallback reasoning.' }])
      })
    ).toEqual(['Canonical reasoning.'])
  })
})
