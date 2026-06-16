import { describe, expect, it } from 'vitest'

import { reasoningPart, textPart } from '@/lib/chat-messages'

import { applyReasoningAvailable } from './use-message-stream'

describe('applyReasoningAvailable', () => {
  it('inserts available reasoning before existing assistant text', () => {
    const parts = applyReasoningAvailable([textPart('Final answer.')], 'Late reasoning.')

    expect(parts.map(part => part.type)).toEqual(['reasoning', 'text'])
    expect(parts[0]).toEqual(reasoningPart('Late reasoning.'))
    expect(parts[1]).toEqual(textPart('Final answer.'))
  })

  it('keeps the current reasoning part when one already exists', () => {
    const existing = [reasoningPart('Streaming reasoning.'), textPart('Final answer.')]
    const parts = applyReasoningAvailable(existing, 'Late reasoning.')

    expect(parts).toBe(existing)
  })
})
