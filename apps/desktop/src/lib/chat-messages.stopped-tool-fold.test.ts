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

  // A host plugin stopped the turn at its tool call, applied a grant and continued the reply
  // with a hidden prompt: the stored rows close the stopped tool tail, then the reply goes on.
  const closing: SessionMessage = {
    id: 4,
    role: 'assistant',
    content: '',
    display_kind: 'hidden',
    timestamp: 4,
    display_metadata: '{"interrupted": true}'
  }

  const stoppedTail = [...rows.slice(0, 2), { ...rows[2], content: '[Command interrupted]' }, closing]

  it('drops the stopped mark when the reply went on after a hidden prompt', () => {
    const continued: SessionMessage[] = [
      ...stoppedTail,
      { id: 5, role: 'user', content: '[Sandbox] Continue where you left off.', display_kind: 'hidden', timestamp: 5 },
      {
        id: 6,
        role: 'assistant',
        content: '',
        timestamp: 6,
        tool_calls: [{ id: 'call-2', type: 'function', function: { name: 'terminal', arguments: '{}' } }]
      },
      { id: 7, role: 'tool', content: 'ok', tool_call_id: 'call-2', timestamp: 7 },
      { id: 8, role: 'assistant', content: 'Here it is.', timestamp: 8 }
    ]

    const replies = toChatMessages(continued).filter(message => message.role === 'assistant')

    expect(chatMessageText(replies[replies.length - 1])).toContain('Here it is.')
    expect(replies.map(message => message.interrupted ?? false)).toEqual(replies.map(() => false))
  })

  it('keeps the stopped mark when the user writes next', () => {
    const userNext: SessionMessage[] = [
      ...stoppedTail,
      { id: 5, role: 'user', content: 'never mind', timestamp: 5 },
      { id: 6, role: 'assistant', content: 'OK.', timestamp: 6 }
    ]

    const replies = toChatMessages(userNext).filter(message => message.role === 'assistant')

    expect(replies.map(message => [chatMessageText(message), message.interrupted ?? false])).toEqual([
      ['Looking.', true],
      ['OK.', false]
    ])
  })
})
