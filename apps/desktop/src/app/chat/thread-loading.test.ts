import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { lastVisibleMessageIsUser, threadLoadingState } from './thread-loading'

function message(id: string, role: ChatMessage['role'], hidden = false): ChatMessage {
  return {
    id,
    role,
    parts: [{ type: 'text', text: `${role}:${id}` }],
    hidden
  }
}

describe('thread loading state', () => {
  it('returns session when routed session is still hydrating', () => {
    expect(threadLoadingState(true, true, true, false)).toBe('session')
  })

  it('returns response while awaiting an assistant reply to the last visible user message', () => {
    const messages = [message('u1', 'user'), message('a1', 'assistant', true)]

    expect(lastVisibleMessageIsUser(messages)).toBe(true)
    expect(threadLoadingState(false, true, true, lastVisibleMessageIsUser(messages))).toBe('response')
  })

  it('does not show response loading when the last visible message is not user-authored', () => {
    const messages = [message('u1', 'user'), message('a1', 'assistant')]

    expect(lastVisibleMessageIsUser(messages)).toBe(false)
    expect(threadLoadingState(false, true, true, lastVisibleMessageIsUser(messages))).toBeUndefined()
  })
})
