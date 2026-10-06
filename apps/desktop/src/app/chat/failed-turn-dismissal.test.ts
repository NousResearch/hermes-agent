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

/** A saved assistant reply (`mergeStoredAssistantErrors` grafted the error
 * onto durable content — rowId present, parts intact). */
const savedReplyWithGraftedError = (id: string, rowId: number): ChatMessage => ({
  id,
  role: 'assistant',
  rowId,
  parts: [{ type: 'text', text: 'saved partial answer' }],
  error: 'Connection error.',
  pending: false
})

describe('clearDismissedErrorRows', () => {
  it('keeps a saved reply with a grafted error, clearing only the error', () => {
    const saved = savedReplyWithGraftedError('assistant-saved', 77)
    const after = answer('answer-next')

    const dismissed = clearDismissedErrorRows([user('stored-user'), saved, after], 'assistant-saved')

    expect(dismissed).toHaveLength(3)
    expect(dismissed.map(message => message.id)).toEqual(['stored-user', 'assistant-saved', 'answer-next'])

    const kept = dismissed[1]
    expect(kept.rowId).toBe(77)
    expect(kept.parts).toEqual(saved.parts)
    expect(kept.error).toBeUndefined()
    expect(kept.errorSurface).toBeUndefined()
    expect(kept.pending).toBe(false)
  })

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
