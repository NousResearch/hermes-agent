import { describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'
import { textPart } from '@/lib/chat-messages'

import { collapseDuplicateFinalAfterToolInterim } from './collapse-duplicate-final'

const tool = (toolCallId: string): ChatMessagePart =>
  ({
    args: {} as never,
    argsText: '{}',
    toolCallId,
    toolName: 'terminal',
    type: 'tool-call' as const
  }) as ChatMessagePart

const assistant = (id: string, parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  parts,
  role: 'assistant',
  ...extra
})

const settle = (message: ChatMessage): ChatMessage => ({ ...message, interim: false, pending: false })

const fold = (messages: ChatMessage[], extra: { finalText?: string; hasFailure?: boolean } = {}) =>
  collapseDuplicateFinalAfterToolInterim(messages, messages.length - 1, {
    completeMessage: settle,
    finalText: extra.finalText ?? 'same reply',
    hasFailure: extra.hasFailure ?? false,
    interimBoundaryPending: true
  })

describe('collapseDuplicateFinalAfterToolInterim', () => {
  it('folds an identical live tool row into the sealed interim', () => {
    const next = fold([
      assistant('interim', [textPart('same reply')], { interim: true }),
      assistant('live', [tool('t1')])
    ])

    expect(next?.keptId).toBe('interim')
    expect(next?.messages).toHaveLength(1)
    expect(next?.messages[0].id).toBe('interim')
    expect(next?.messages[0].interim).toBe(false)
    expect(next?.messages[0].parts.filter(part => part.type === 'text')).toHaveLength(1)
    expect(next?.messages[0].parts.filter(part => part.type === 'tool-call')).toHaveLength(1)
  })

  it('returns null when the live row streamed a distinct body', () => {
    expect(
      fold([
        assistant('interim', [textPart('same reply')], { interim: true }),
        assistant('live', [textPart('a different live body'), tool('t1')])
      ])
    ).toBeNull()
  })

  it('returns null on a failure frame', () => {
    expect(
      fold(
        [assistant('interim', [textPart('same reply')], { interim: true }), assistant('live', [tool('t1')])],
        { hasFailure: true }
      )
    ).toBeNull()
  })

  it('returns null when the prior interim text is distinct', () => {
    expect(
      fold([
        assistant('interim', [textPart('Let me inspect that.')], { interim: true }),
        assistant('live', [tool('t1')])
      ])
    ).toBeNull()
  })

  it('renames colliding tool ids inside the merged message', () => {
    const next = fold([
      assistant('interim', [textPart('same reply'), tool('terminal_0')], { interim: true }),
      assistant('live', [tool('terminal_0')])
    ])

    const ids = next?.messages[0].parts
      .filter((part): part is Extract<ChatMessagePart, { type: 'tool-call' }> => part.type === 'tool-call')
      .map(part => part.toolCallId)

    expect(ids).toHaveLength(2)
    expect(new Set(ids).size).toBe(2)
  })
})
