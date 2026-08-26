import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { clearDismissedErrorRows } from './failed-turn-dismissal'

const user = (id: string, rowId?: number): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text: 'prompt' }],
  ...(rowId === undefined ? {} : { rowId })
})

const error = (id: string, parts: ChatMessage['parts'] = []): ChatMessage => ({
  id,
  role: 'assistant',
  parts,
  error: 'Connection error.',
  pending: false
})

const answer = (id: string): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'text', text: 'answer' }]
})

describe('clearDismissedErrorRows', () => {
  it('removes a bare error and its optimistic companion user row', () => {
    const messages = [user('user-1723000000000-abc123'), error('failed'), user('stored-next'), answer('answer-next')]

    expect(clearDismissedErrorRows(messages, 'failed').map(message => message.id)).toEqual([
      'stored-next',
      'answer-next'
    ])
  })

  it('removes partial reasoning, text, and tool payload with the failed assistant row', () => {
    const messages = [
      user('user-1723000000000-def456'),
      error('failed', [
        { type: 'reasoning', text: 'thought' },
        { type: 'text', text: 'partial result' },
        {
          type: 'tool-call',
          toolCallId: 'call-1',
          toolName: 'read_file',
          result: 'partial'
        } as ChatMessage['parts'][number]
      ])
    ]

    expect(clearDismissedErrorRows(messages, 'failed')).toEqual([])
  })

  it('preserves an authoritative companion user row', () => {
    const authoritative = user('user-1723000000000-ghi789', 101)

    expect(clearDismissedErrorRows([authoritative, error('failed')], 'failed')).toEqual([authoritative])
  })

  it('preserves a non-optimistic companion user row', () => {
    expect(
      clearDismissedErrorRows([user('stored-user'), error('failed')], 'failed').map(message => message.id)
    ).toEqual(['stored-user'])
  })

  it('ignores a targeted non-assistant error row', () => {
    const erroredUser = { ...user('user-1723000000000-jkl012'), error: 'client marker' }

    expect(clearDismissedErrorRows([erroredUser], erroredUser.id)).toEqual([erroredUser])
  })

  it('preserves array identity when no failed assistant matches', () => {
    const messages = [user('stored-user'), answer('answer')]

    expect(clearDismissedErrorRows(messages, 'missing')).toBe(messages)
  })
})
