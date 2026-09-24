import { describe, expect, it } from 'vitest'
import { appendReasoningPart, assistantTextPart, reasoningPart, toChatMessages, upsertToolPart } from './chat-messages'

const toolCallPart = (id: string) =>
  upsertToolPart([], { tool_id: id, name: 'terminal', args: {}, result: 'ok' }, 'complete')[0]

const items = [
  {
    type: 'reasoning',
    id: 'rs',
    summary: [
      { type: 'summary_text', text: 'Inspect' },
      { type: 'summary_text', text: 'Check' }
    ]
  }
]

describe('native reasoning identity without a second commentary projector (#79727)', () => {
  it('uses only backend-authorized identities matching all displayed reasoning', () => {
    const row = {
      role: 'assistant' as const,
      content: 'Done.',
      reasoning: 'Inspect\nCheck\n\nPublic.',
      display_reasoning: 'Inspect\nCheck',
      display_commentary: ['Public.'],
      display_reasoning_items: items,
      codex_reasoning_items: [
        { type: 'reasoning', id: 'private', summary: [{ type: 'summary_text', text: 'Not authorized' }] }
      ]
    }
    const [message] = toChatMessages([row])
    expect(message.parts.filter(part => part.type === 'reasoning').map(part => [part.sourceId, part.text])).toEqual([
      ['rs:summary:0', 'Inspect'],
      ['rs:summary:1', 'Check']
    ])
    expect(message.parts.filter(part => part.type === 'text').map(part => part.text)).toEqual(['Public.', 'Done.'])
    const [fallback] = toChatMessages([{ ...row, display_reasoning_items: undefined }])
    expect(fallback.parts.filter(part => part.type === 'reasoning')).toEqual([reasoningPart('Inspect\nCheck')])
    const [mismatch] = toChatMessages([{ ...row, display_reasoning: 'Different' }])
    expect(mismatch.parts.filter(part => part.type === 'reasoning')).toEqual([reasoningPart('Different')])
  })

  it('never joins native sources across text, tools, completion, or legacy chunks', () => {
    const first = reasoningPart('Before', 1, 'rs')
    for (const boundary of [assistantTextPart('Public', 2), toolCallPart('tc')]) {
      expect(appendReasoningPart([first, boundary], 'After', 3, 'rs')).toEqual([
        first,
        boundary,
        reasoningPart('After', 3, 'rs')
      ])
    }
    expect(appendReasoningPart([{ ...first, completedAt: 2 }], 'Later', 3, 'rs')).toHaveLength(2)
    expect(appendReasoningPart([first], 'Legacy', 3)).toEqual([first, reasoningPart('Legacy', 3)])
    expect(appendReasoningPart([first], ' continuation', 2, 'rs')).toEqual([
      reasoningPart('Before continuation', 1, 'rs')
    ])
  })
})
