import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { dropLiveRowsRepresentedByCommitted } from '@/lib/live-row-dedupe'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

describe('dropLiveRowsRepresentedByCommitted', () => {
  // The app log's own proven pair: a committed row of 240 characters beside a
  // live row of 242, the live row still carrying pending=true / status=running
  // minutes after the turn ended (committed row: status=complete:stop).
  it('drops a still-pending live row whose answer the committed rows already carry', () => {
    const messages = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('1790534509.9796042-116-assistant', 'assistant', 'Coverage is over the floor.', { rowId: 23169 }),
      msg('assistant-stream-1790534521580-28', 'assistant', 'Coverage is over the floor.\n', { pending: true })
    ]

    expect(dropLiveRowsRepresentedByCommitted(messages).map(message => message.id)).toEqual([
      '3-user',
      '1790534509.9796042-116-assistant'
    ])
  })

  it('drops a settled live row whose answer the committed rows already carry', () => {
    const messages = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('1790534509.9796042-116-assistant', 'assistant', 'Coverage is over the floor.', { rowId: 116 }),
      msg('assistant-stream-1790534521580-28', 'assistant', 'Coverage is over the floor.\n', { pending: false })
    ]

    expect(dropLiveRowsRepresentedByCommitted(messages).map(message => message.id)).toEqual([
      '3-user',
      '1790534509.9796042-116-assistant'
    ])
  })

  // The running turn's live tail must survive: it is still gaining text, so its
  // fold is not equal to the committed row's - it extends it.
  it('keeps a streaming live row that runs past the committed answer', () => {
    const messages = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('1790534509.9796042-116-assistant', 'assistant', 'Coverage is over the floor.', { rowId: 116 }),
      msg('assistant-stream-1790534521580-28', 'assistant', 'Coverage is over the floor. Checking the rest.', {
        pending: true
      })
    ]

    expect(dropLiveRowsRepresentedByCommitted(messages).map(message => message.id)).toEqual([
      '3-user',
      '1790534509.9796042-116-assistant',
      'assistant-stream-1790534521580-28'
    ])
  })

  it('keeps a live row that says something the committed rows do not', () => {
    const messages = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('1790534509.9796042-116-assistant', 'assistant', 'The earlier answer.', { rowId: 116 }),
      msg('assistant-stream-1790534521580-28', 'assistant', 'The newer answer.', { pending: false })
    ]

    expect(dropLiveRowsRepresentedByCommitted(messages).map(message => message.id)).toEqual([
      '3-user',
      '1790534509.9796042-116-assistant',
      'assistant-stream-1790534521580-28'
    ])
  })

  // completion-boundaries.test.tsx guards this shape: an identical prompt
  // submitted again is a NEW occurrence, so its live row may repeat the previous
  // answer word for word and must survive - the committed twin sits before the
  // second user row, so it belongs to the previous turn.
  it('keeps a live row whose identical text belongs to a new turn', () => {
    const messages = [
      msg('1-user', 'user', 'Give the answer.', { rowId: 1 }),
      msg('2-assistant', 'assistant', 'The answer is unchanged.', { rowId: 2 }),
      msg('3-user', 'user', 'Give the answer.', { rowId: 3 }),
      msg('assistant-stream-1790534521580-28', 'assistant', 'The answer is unchanged.', { pending: true })
    ]

    expect(dropLiveRowsRepresentedByCommitted(messages).map(message => message.id)).toEqual([
      '1-user',
      '2-assistant',
      '3-user',
      'assistant-stream-1790534521580-28'
    ])
  })

  it('returns the same array by reference when nothing is dropped', () => {
    const messages = [msg('3-user', 'user', 'prompt b', { rowId: 3 }), msg('4-assistant', 'assistant', 'Done.', { rowId: 4 })]

    expect(dropLiveRowsRepresentedByCommitted(messages)).toBe(messages)
  })
})
