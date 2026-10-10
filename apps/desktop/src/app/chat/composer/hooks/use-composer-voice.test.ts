import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { activeToolLabelFromMessages } from './use-composer-voice'

function assistantMessage(parts: ChatMessage['parts'], overrides: Partial<ChatMessage> = {}): ChatMessage {
  return { id: 'm1', parts, role: 'assistant', ...overrides }
}

describe('activeToolLabelFromMessages', () => {
  it('reports a running tool call (no result, not completed)', () => {
    const messages = [assistantMessage([{ args: {}, toolCallId: 'call-1', toolName: 'terminal', type: 'tool-call' }])]

    expect(activeToolLabelFromMessages(messages)).toBe('terminal')
  })

  it('does not report a call sealed without a result as running (turn stopped, completion event lost)', () => {
    // completeOpenTimelineParts stamps `completedAt` on a still-open part when the turn settles,
    // without ever supplying `result` — the call is dead, not running (matches ToolFallback's
    // settledWithoutResult guard in message-parts.tsx).
    const messages = [
      assistantMessage([
        { args: {}, completedAt: 42, toolCallId: 'call-1', toolName: 'terminal', type: 'tool-call' }
      ])
    ]

    expect(activeToolLabelFromMessages(messages)).toBeNull()
  })

  it('does not report a settled call with a result as running', () => {
    const messages = [
      assistantMessage([
        { args: {}, completedAt: 42, result: 'ok', toolCallId: 'call-1', toolName: 'terminal', type: 'tool-call' }
      ])
    ]

    expect(activeToolLabelFromMessages(messages)).toBeNull()
  })

  it('uses only the last visible assistant turn, ignoring hidden and non-assistant messages', () => {
    const messages: ChatMessage[] = [
      assistantMessage([{ args: {}, toolCallId: 'call-0', toolName: 'old-tool', type: 'tool-call' }], { id: 'm0' }),
      { id: 'u1', parts: [], role: 'user' },
      assistantMessage([{ args: {}, toolCallId: 'call-h', toolName: 'hidden-tool', type: 'tool-call' }], {
        hidden: true,
        id: 'mh'
      }),
      assistantMessage([{ args: {}, toolCallId: 'call-2', toolName: 'current-tool', type: 'tool-call' }], {
        id: 'm2'
      })
    ]

    expect(activeToolLabelFromMessages(messages)).toBe('current-tool')
  })

  it('returns null with no messages or no assistant tool-call parts', () => {
    expect(activeToolLabelFromMessages([])).toBeNull()
    expect(activeToolLabelFromMessages([assistantMessage([{ text: 'hello', type: 'text' }])])).toBeNull()
  })
})
