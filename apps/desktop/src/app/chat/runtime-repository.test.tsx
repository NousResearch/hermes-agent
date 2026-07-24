import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { type ChatMessage, getChatMessageListUpdate, updateChatMessageAt } from '@/lib/chat-messages'

import { runtimeMessageRepositoryDelta, useRuntimeMessageRepository } from './runtime-repository'

function message(id: string, role: ChatMessage['role'], text: string, pending = false): ChatMessage {
  return {
    id,
    role,
    parts: [{ type: 'text', text }],
    pending
  }
}

describe('useRuntimeMessageRepository', () => {
  it('requires the exact predecessor when updates are batched', () => {
    const messages = [message('user-1', 'user', 'Question'), message('assistant-1', 'assistant', 'A', true)]
    const first = updateChatMessageAt(messages, 1, current => message(current.id, current.role, 'AB', true))
    const second = updateChatMessageAt(first, 1, current => message(current.id, current.role, 'ABC', true))

    expect(getChatMessageListUpdate(messages, second)).toBeNull()
    expect(getChatMessageListUpdate(first, second)).toMatchObject({
      index: 1,
      previousMessage: first[1],
      message: second[1]
    })
    expect(getChatMessageListUpdate(messages, first)).not.toBeNull()
  })

  it('preserves settled runtime message identity when only the streaming tail changes', () => {
    const user = message('user-1', 'user', 'Question')
    const settled = message('assistant-1', 'assistant', 'Settled answer')
    const streaming = message('assistant-2', 'assistant', 'A', true)

    const { result, rerender } = renderHook(
      ({ messages }: { messages: ChatMessage[] }) => useRuntimeMessageRepository(messages),
      { initialProps: { messages: [user, settled, streaming] } }
    )

    const initialRepository = result.current
    const nextStreaming = message('assistant-2', 'assistant', 'AB', true)

    rerender({ messages: [user, settled, nextStreaming] })

    expect(result.current.messages[0]?.message).toBe(initialRepository.messages[0]?.message)
    expect(result.current.messages[1]?.message).toBe(initialRepository.messages[1]?.message)
    expect(result.current.messages[2]?.message).not.toBe(initialRepository.messages[2]?.message)
    expect(result.current.messages.map(item => item.parentId)).toEqual([null, 'user-1', 'assistant-1'])
    expect(result.current.headId).toBe('assistant-2')
  })

  it('forwards an annotated pending-tail update without rebuilding settled repository items', () => {
    const messages = [
      message('user-1', 'user', 'Question'),
      message('assistant-1', 'assistant', 'Settled answer'),
      message('assistant-2', 'assistant', 'A', true)
    ]

    const { result, rerender } = renderHook(
      ({ messages: nextMessages }: { messages: ChatMessage[] }) => useRuntimeMessageRepository(nextMessages),
      { initialProps: { messages } }
    )

    const initialRepository = result.current
    const nextMessages = updateChatMessageAt(messages, 2, current => message(current.id, current.role, 'AB', true))

    rerender({ messages: nextMessages })

    expect(result.current.messages).toBe(initialRepository.messages)
    expect(result.current[runtimeMessageRepositoryDelta]?.message.content).toEqual([{ type: 'text', text: 'AB' }])
    expect(result.current[runtimeMessageRepositoryDelta]?.parentId).toBe('assistant-1')
  })

  it('falls back to a complete repository when the tail settles', () => {
    const messages = [message('user-1', 'user', 'Question'), message('assistant-1', 'assistant', 'A', true)]

    const { result, rerender } = renderHook(
      ({ messages: nextMessages }: { messages: ChatMessage[] }) => useRuntimeMessageRepository(nextMessages),
      { initialProps: { messages } }
    )

    const initialRepository = result.current
    const nextMessages = updateChatMessageAt(messages, 1, current => message(current.id, current.role, 'Done', false))

    rerender({ messages: nextMessages })

    expect(result.current.messages).not.toBe(initialRepository.messages)
    expect(result.current[runtimeMessageRepositoryDelta]).toBeUndefined()
    expect(result.current.messages.at(-1)?.message.status).toEqual({ type: 'complete', reason: 'stop' })
  })
})
