import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { reconcileResumeMessages } from './utils'

describe('resume preserves the position of a pending question', () => {
  it('keeps explanation before clarify across flat snapshots and repeated restoration', () => {
    const prompt = 'Before proceeding, I need your answer:'

    const cached: ChatMessage[] = [
      {
        id: 'a1',
        role: 'assistant',
        pending: true,
        parts: [
          { type: 'text', text: prompt },
          {
            type: 'tool-call',
            toolName: 'clarify',
            toolCallId: 'question-1',
            args: { questions: [{ question: 'Continue?', choices: ['Yes', 'No'] }] }
          }
        ]
      }
    ]

    for (const text of [prompt, prompt + ' Please choose.']) {
      const flat: ChatMessage[] = [{ id: 'a1', role: 'assistant', pending: true, parts: [{ type: 'text', text }] }]
      let restored = reconcileResumeMessages(flat, cached)
      restored = reconcileResumeMessages(flat, restored)
      expect(restored[0].parts.map(part => part.type)).toEqual(['text', 'tool-call'])
      expect(restored[0].parts[0]).toMatchObject({ text })
      expect(restored[0].parts[1]).toMatchObject({ toolCallId: 'question-1' })
    }
  })

  it('keeps interleaved activity in place when a flat dump is not an answer extension', () => {
    const cached: ChatMessage[] = [
      {
        id: 'assistant-stream-1',
        role: 'assistant',
        pending: true,
        parts: [
          { type: 'text', text: 'Checking.' },
          { type: 'tool-call', toolName: 'terminal', toolCallId: 'check-1', result: 'ok' },
          { type: 'text', text: 'Choose:' },
          { type: 'tool-call', toolName: 'clarify', toolCallId: 'question-1', args: { question: 'Continue?' } }
        ]
      }
    ]

    const flat: ChatMessage[] = [
      {
        id: 'assistant-stream-1',
        role: 'assistant',
        pending: true,
        parts: [{ type: 'text', text: 'Flattened activity dump' }]
      }
    ]

    const [restored] = reconcileResumeMessages(flat, cached)
    expect(restored.parts).toEqual(cached[0].parts)
  })
})
