import { describe, expect, it } from 'vitest'

import type { ChatMessagePart } from '@/lib/chat-messages'

import { canStartWithConnections } from './first-build-start'

const wait = (overrides: Partial<Extract<ChatMessagePart, { type: 'tool-call' }>>): ChatMessagePart =>
  ({
    type: 'tool-call',
    toolCallId: 'wait-1',
    toolName: 'manage_connections',
    args: { action: 'wait' },
    ...overrides
  }) as ChatMessagePart

describe('canStartWithConnections', () => {
  it('offers the start while a connection wait is in flight', () => {
    expect(canStartWithConnections(wait({}))).toBe(true)
  })

  it('withdraws the start once the wait is sealed without a result', () => {
    expect(canStartWithConnections(wait({ completedAt: 5 }))).toBe(false)
  })
})
