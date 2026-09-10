import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { setActiveSessionId, setMessages } from '@/store/session'

const mocks = vi.hoisted(() => ({
  ambientRequest: vi.fn(),
  notifyError: vi.fn(),
  requestForOwnedSession: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: () => ({ request: mocks.ambientRequest })
}))

vi.mock('@/store/notifications', () => ({ notifyError: mocks.notifyError }))

vi.mock('@/store/session-states', () => ({
  requestForOwnedSession: mocks.requestForOwnedSession
}))

const { toggleMessageReaction } = await import('@/store/reactions')

describe('toggleMessageReaction owner routing', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setActiveSessionId('stored-bot-session')
    setMessages([
      {
        id: 'message-1',
        role: 'assistant',
        rowId: 42,
        content: 'done',
        parts: [{ type: 'text', text: 'done' }]
      } as ChatMessage
    ])
    mocks.requestForOwnedSession.mockResolvedValue({
      row_id: 42,
      reactions: [{ at: 1_700_000_000, author: 'user', emoji: '👍' }]
    })
  })

  it('dispatches through the durable session owner instead of the ambient active gateway', async () => {
    const message = {
      id: 'message-1',
      role: 'assistant',
      rowId: 42,
      content: 'done',
      parts: [{ type: 'text', text: 'done' }]
    } as ChatMessage

    await toggleMessageReaction(message, '👍')

    expect(mocks.requestForOwnedSession).toHaveBeenCalledTimes(1)
    expect(mocks.requestForOwnedSession).toHaveBeenCalledWith(
      'stored-bot-session',
      expect.any(Function),
      'message.react',
      {
        session_id: 'stored-bot-session',
        row_id: 42,
        emoji: '👍',
        author: 'user'
      }
    )
    expect(mocks.ambientRequest).not.toHaveBeenCalled()
    expect(mocks.notifyError).not.toHaveBeenCalled()
  })
})
