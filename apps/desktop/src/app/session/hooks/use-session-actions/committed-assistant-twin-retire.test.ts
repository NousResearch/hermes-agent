import { describe, expect, it } from 'vitest'

import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart } from '@/lib/chat-messages'

import { preserveLocalPendingTurnMessages } from './utils'

const tool = (id: string): ChatMessagePart =>
  ({ type: 'tool-call', toolCallId: id, toolName: 'terminal', result: 'done' }) as ChatMessagePart

const message = (
  id: string,
  role: ChatMessage['role'],
  parts: ChatMessagePart[],
  extra: Partial<ChatMessage> = {}
): ChatMessage => ({ id, role, parts, ...extra }) as ChatMessage

// #131500: a reconnect restored the session from the durable store mid-turn,
// so the renderer's in-memory list still holds the live copy (pending, with
// the turn's tool calls) while the durable refresh delivers the committed
// reply under a different, positionally synthesized id. Ordinal pairing and
// the full-text guards both miss, so both copies render until a restart.
describe('preserveLocalPendingTurnMessages — committed twin of the live tail (#131500)', () => {
  it('retires the live copy when the durable refresh carries its committed twin sharing the turn tool call', () => {
    const previous = [
      message('user-live', 'user', [textPart('check the model')]),
      message('assistant-hidden', 'assistant', [textPart('earlier hidden directive')], { hidden: true }),
      message('assistant-stream-live', 'assistant', [tool('call-1'), textPart('Model is up.')], { pending: true })
    ]

    // History folds the turn's narration, tool round and answer into one
    // committed bubble under a fresh id.
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-stored', 'assistant', [textPart('Working on it.'), tool('call-1'), textPart('Model is up.')], {
        rowId: 196701
      })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toEqual(next)
    expect(merged.filter(row => chatMessageText(row).includes('Model is up.'))).toHaveLength(1)
  })

  it('retires the live copy when the committed twin differs only in folded whitespace', () => {
    const previous = [
      message('user-live', 'user', [textPart('check the model')]),
      message('assistant-hidden', 'assistant', [textPart('earlier hidden directive')], { hidden: true }),
      message('assistant-stream-live', 'assistant', [textPart('Model is  up.')], { pending: true })
    ]

    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-stored', 'assistant', [textPart('Model is up.')], { rowId: 196701 })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged).toEqual(next)
    expect(merged.map(row => row.id)).toEqual(['user-stored', 'assistant-stored'])
  })

  it('keeps the live copy when the committed fold lags behind the stream', () => {
    const previous = [
      message('user-live', 'user', [textPart('check the model')]),
      message('assistant-hidden', 'assistant', [textPart('earlier hidden directive')], { hidden: true }),
      // The narration sealed into its own interim row, so the pending live row
      // holds only the answer; the store flushed just the narration + tool
      // round so far — its fold has no answer after the last tool.
      message('assistant-stream-live', 'assistant', [tool('call-1'), textPart('Model is up.')], { pending: true })
    ]

    // The store flushed only the narration so far; the streamed answer the
    // local row holds is the only copy of it.
    const next = [
      message('user-stored', 'user', [textPart('check the model')], { rowId: 196700 }),
      message('assistant-stored', 'assistant', [textPart('Working on it.'), tool('call-1')], { rowId: 196701 })
    ]

    const merged = preserveLocalPendingTurnMessages(next, previous)

    expect(merged.map(row => row.id)).toContain('assistant-stream-live')
  })
})
