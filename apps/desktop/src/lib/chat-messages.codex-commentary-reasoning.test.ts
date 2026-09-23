import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from './chat-messages'

// Persisted tool-call row where the Responses adapter flattened public
// `phase=commentary` text into the `reasoning` blob beside genuine reasoning
// summaries. Hydration must keep the public sentence in exactly one channel
// (assistant text) and leave private reasoning in Thinking.
const commentary = 'Checking the migration files now.'
const reasoningBlob = `Plan summary\nApproach overview\n${commentary}`

const toolCallRow: SessionMessage = {
  id: 501,
  role: 'assistant',
  content: '',
  reasoning: reasoningBlob,
  reasoning_content: reasoningBlob,
  codex_message_items: [
    {
      type: 'message',
      id: 'msg_commentary',
      role: 'assistant',
      phase: 'commentary',
      status: 'completed',
      content: [{ type: 'output_text', text: commentary }]
    }
  ],
  tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }],
  timestamp: 2
}

const reasoningText = (row: SessionMessage) =>
  toChatMessages([row])[0]
    ?.parts.filter(part => part.type === 'reasoning')
    .map(part => part.text)
    .join('\n\n') ?? ''

describe('codex commentary flattened into reasoning', () => {
  it('shows commentary as assistant text and removes it from Thinking', () => {
    const [message] = toChatMessages([toolCallRow])

    expect(chatMessageText(message)).toBe(commentary)
    expect(reasoningText(toolCallRow)).toContain('Plan summary')
    expect(reasoningText(toolCallRow)).toContain('Approach overview')
    expect(reasoningText(toolCallRow)).not.toContain(commentary)
  })

  it('accepts the REST JSON-string sidecar identically', () => {
    const viaJson: SessionMessage = {
      ...toolCallRow,
      codex_message_items: JSON.stringify(toolCallRow.codex_message_items)
    }

    const [message] = toChatMessages([viaJson])

    expect(chatMessageText(message)).toBe(commentary)
    expect(reasoningText(viaJson)).not.toContain(commentary)
  })

  it('keeps private analysis in Thinking and out of assistant text', () => {
    const row: SessionMessage = {
      ...toolCallRow,
      codex_message_items: [
        ...(toolCallRow.codex_message_items as unknown[]),
        {
          type: 'message',
          id: 'msg_analysis',
          role: 'assistant',
          phase: 'analysis',
          status: 'completed',
          content: [{ type: 'output_text', text: 'Scratchpad thoughts.' }]
        }
      ],
      reasoning: `${reasoningBlob}\nScratchpad thoughts.`
    }

    const [message] = toChatMessages([row])

    expect(chatMessageText(message)).toBe(commentary)
    expect(chatMessageText(message)).not.toContain('Scratchpad thoughts.')
    expect(reasoningText(row)).toContain('Scratchpad thoughts.')
  })

  it('does not duplicate commentary already persisted as canonical content', () => {
    const row: SessionMessage = { ...toolCallRow, content: commentary }
    const [message] = toChatMessages([row])
    const texts = message.parts.filter(part => part.type === 'text').map(part => part.text)

    expect(texts).toEqual([commentary])
    expect(reasoningText(row)).not.toContain(commentary)
  })
})
