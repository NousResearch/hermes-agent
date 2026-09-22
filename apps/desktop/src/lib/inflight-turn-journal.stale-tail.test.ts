import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { mergeInFlightMessages } from '@/lib/inflight-turn-journal'

function user(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }], ...extra }
}

function assistant(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }], ...extra }
}

/** A journal snapshot only ever describes the turn that was streaming. Once the
 *  transcript holds that reply AND a newer committed turn, replaying the
 *  snapshot puts an already committed reply below newer content (#70108). */
describe('stale journal tail against superseded turns', () => {
  const base: ChatMessage[] = [
    user('u1', 'first prompt'),
    assistant('a1', 'first reply', { rowId: 101, durableComplete: true }),
    user('u2', 'second prompt'),
    assistant('a2', 'second reply', { rowId: 102, durableComplete: true })
  ]

  it('retires a stale tail whose prompt row no longer matches the transcript', () => {
    // Snapshot taken while turn 1 streamed: the reply row carries no durable id
    // yet, and its prompt row drifted from the transcript (rewritten prompt or a
    // changed attachment signature), so no base user matches it.
    const tail: ChatMessage[] = [
      user('j1', 'first prompt', { attachmentRefs: ['@file:notes.md'] }),
      assistant('live-1', 'first reply')
    ]

    const recovered = mergeInFlightMessages(base, tail, { keepPending: false })

    expect(recovered.caughtUp).toBe(true)
    expect(recovered.messages).toBe(base)
  })

  it('still appends a tail the transcript never committed', () => {
    const tail: ChatMessage[] = [user('j2', 'third prompt'), assistant('live-2', 'third reply')]

    const recovered = mergeInFlightMessages(base, tail, { keepPending: false })

    expect(recovered.applied).toBe(true)
    expect(recovered.messages.map(message => message.id)).toEqual([
      ...base.map(message => message.id),
      'j2',
      'live-2'
    ])
  })
})
