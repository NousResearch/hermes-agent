import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from './chat-messages'

// A turn stopped after a tool round is stored as narration + tool call, the
// tool result, then the interrupted partial. The partial folds into the tool
// bubble on read; the stored copy that replaces the live one (any refresh)
// must still say the reply was stopped.
describe('stopped reply folded into its tool bubble', () => {
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: 'look it up', timestamp: 1 },
    {
      id: 2,
      role: 'assistant',
      content: 'Looking.',
      timestamp: 2,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }]
    },
    { id: 3, role: 'tool', content: 'ok', tool_call_id: 'call-1', timestamp: 3 },
    { id: 4, role: 'assistant', content: 'The answer is', timestamp: 4, display_metadata: '{"interrupted": true}' }
  ]

  it('keeps the stopped mark of the interrupted row it absorbs', () => {
    const replies = toChatMessages(rows).filter(message => message.role === 'assistant')

    expect(replies).toHaveLength(1)
    expect(chatMessageText(replies[0])).toContain('The answer is')
    expect(replies[0].interrupted).toBe(true)
  })

  it('keeps the stopped mark of a turn stopped during its tool run', () => {
    // Stored by the backend's close of an interrupted tool tail: no reply, an empty hidden row.
    const closing: SessionMessage = {
      id: 4,
      role: 'assistant',
      content: '',
      display_kind: 'hidden',
      timestamp: 4,
      display_metadata: '{"interrupted": true}'
    }

    const stoppedInTool = [...rows.slice(0, 2), { ...rows[2], content: '[Command interrupted]' }, closing]
    const replies = toChatMessages(stoppedInTool).filter(message => message.role === 'assistant')

    expect(replies.map(message => [chatMessageText(message), message.interrupted])).toEqual([['Looking.', true]])
  })

  it('leaves a finished tool turn unmarked', () => {
    const finished = rows.map(row => (row.id === 4 ? { ...row, display_metadata: undefined } : row))
    const replies = toChatMessages(finished).filter(message => message.role === 'assistant')

    expect(replies.map(message => message.interrupted)).toEqual([undefined])
  })
})
