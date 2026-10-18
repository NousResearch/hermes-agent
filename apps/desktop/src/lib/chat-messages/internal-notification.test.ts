import { describe, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { toChatMessages } from '@/lib/chat-messages/hydration'

const WAKE =
  '[kanban] Task t_3fc68611 completed.\nTitle: Quiet chats\nAssignee: @hermes\nBoard: default\n\nCheck the result or decide the next step.\n\nThis is an automatic task-status notification.'

describe('internal_notification rows', () => {
  it('render as a system board-update notice, never as the user speaking', () => {
    const [row, reply] = toChatMessages([
      { role: 'user', content: WAKE, display_kind: 'internal_notification' },
      { role: 'assistant', content: 'Shipped: the chat is quiet now.' }
    ])

    expect(row.role).toBe('system')
    expect(chatMessageText(row)).toBe('Board update · t_3fc68611 completed · Quiet chats')
    // The full wake text stays one click away, never in the label.
    expect(row.asyncResult).toContain('automatic task-status notification')
    expect(reply.role).toBe('assistant')
  })

  it('backend-hidden plumbing rows (silent wake tool rounds, bare NO_REPLY) paint nothing', () => {
    const messages = toChatMessages([
      { role: 'user', content: 'hello' },
      { role: 'user', content: WAKE, display_kind: 'hidden' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'c1', function: { name: 'kanban_show', arguments: '{}' } }],
        display_kind: 'hidden'
      },
      { role: 'tool', content: '{}', tool_call_id: 'c1', display_kind: 'hidden' },
      { role: 'assistant', content: 'NO_REPLY', display_kind: 'hidden' }
    ])

    expect(messages.map(m => [m.role, chatMessageText(m)])).toEqual([['user', 'hello']])
  })
})
