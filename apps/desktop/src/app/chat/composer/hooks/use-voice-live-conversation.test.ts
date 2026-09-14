// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'

import { chunkForCommentary, toLiveHistory } from '@/lib/voice-live'

import { delegationPrompt } from './use-voice-live-conversation'

describe('GPT-Live delegation → Hermes turn', () => {
  it('sends the latest user words as the turn and the exchange as model-only context', () => {
    // The delegation event carries no text: both are reconstructed from
    // transcript deltas, fragments of one speaker concatenated as received.
    const { context, prompt } = delegationPrompt([
      { endMs: 1000, speaker: 'assistant', startMs: 0, text: 'Hi, how ' },
      { endMs: 1500, speaker: 'assistant', startMs: 1000, text: 'can I help?' },
      { endMs: 2500, speaker: 'user', startMs: 1500, text: 'What is ' },
      { endMs: 3200, speaker: 'user', startMs: 2500, text: 'the weather in Paris?' }
    ])

    expect(prompt).toBe('What is the weather in Paris?')
    expect(context).toContain('Voice assistant: Hi, how can I help?')
    expect(context).toContain('User: What is the weather in Paris?')
  })

  it('splits a long reply into vendor-sized commentary appends on sentence boundaries', () => {
    const sentence = 'This is a sentence about the result. '
    const chunks = chunkForCommentary(sentence.repeat(80), 400)

    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every(chunk => chunk.length <= 400)).toBe(true)
    expect(chunks.every(chunk => chunk.endsWith('.'))).toBe(true)
    expect(chunks.join(' ')).toBe(sentence.repeat(80).trim())
  })

  it('seeds the live session with the most recent text turns within budget', () => {
    const turns = Array.from({ length: 40 }, (_, index) => ({
      role: (index % 2 === 0 ? 'user' : 'assistant') as 'assistant' | 'user',
      text: `turn ${index}`
    }))

    const history = toLiveHistory(turns, 6)

    expect(history).toHaveLength(6)
    expect(history.at(-1)?.content[0]?.text).toBe('turn 39')
    expect(history[0]?.role).toBe('user')
    expect(history.find(m => m.role === 'assistant')?.content[0]?.type).toBe('output_text')
  })
})
