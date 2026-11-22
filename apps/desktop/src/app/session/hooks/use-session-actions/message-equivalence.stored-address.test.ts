import { describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'

import { chatMessagesEquivalent } from './message-equivalence'

// A live part that names its stored row gives way to the page's copy of that row
// (lib/chat-messages/live-row-identity). A comparison that called an unaddressed copy
// equal to the addressed one would keep the copy that can never give way.
describe('a part’s stored address in the transcript comparisons', () => {
  const reply = (part: ChatMessagePart) => ({ id: 'assistant-reply', parts: [part], role: 'assistant' }) as ChatMessage
  const text: ChatMessagePart = { type: 'text', text: 'looking' }
  const call: ChatMessagePart = { type: 'tool-call', toolCallId: 'call_0', toolName: 'terminal', args: {} }

  it.each([
    ['a text part’s row', text, { sourceRowId: 7 }],
    ['a text part’s uid', text, { sourceMessageUid: 'uid-7' }],
    ['a call’s row', call, { sourceCallRowId: 7 }],
    ['a call’s uid', call, { sourceMessageUid: 'uid-7' }]
  ] as const)('is a change: %s', (_, part, address) => {
    const named = reply({ ...part, ...address } as ChatMessagePart)

    expect(chatMessagesEquivalent(reply(part), named)).toBe(false)
    expect(chatMessagesEquivalent(named, reply({ ...part, ...address } as ChatMessagePart))).toBe(true)
  })
})
