import { expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from './chat-messages'

it('hydrates the SQLite JSON-text sidecar returned by REST like the decoded RPC sidecar', () => {
  const items = [
    {
      type: 'message',
      role: 'assistant',
      phase: 'final_answer',
      content: [{ type: 'output_text', text: 'Durable reply' }]
    }
  ]

  const row = {
    id: 1,
    role: 'assistant' as const,
    content: '',
    timestamp: 1,
    reasoning: 'Thinking before tool',
    tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }]
  }

  const decoded = toChatMessages([{ ...row, codex_message_items: items }])
  const stored = toChatMessages([{ ...row, codex_message_items: JSON.stringify(items) }])

  expect(stored).toEqual(decoded)
  expect(chatMessageText(stored[0])).toBe('Durable reply')
  const empty = { ...row, reasoning: null, tool_calls: undefined }
  expect(toChatMessages([{ ...empty, codex_message_items: '{invalid' }])).toEqual([])
  expect(toChatMessages([{ ...empty, display_kind: 'hidden', codex_message_items: JSON.stringify(items) }])).toEqual([])

  for (const phase of ['analysis', 'commentary']) {
    expect(toChatMessages([{ ...empty, codex_message_items: JSON.stringify([{ ...items[0], phase }]) }])).toEqual([])
  }

  expect(
    chatMessageText(
      toChatMessages([{ ...row, content: 'Canonical reply', codex_message_items: JSON.stringify(items) }])[0]
    )
  ).toBe('Canonical reply')
})
