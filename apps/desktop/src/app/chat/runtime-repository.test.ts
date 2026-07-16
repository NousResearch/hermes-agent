// @vitest-environment jsdom
import { MessageRepository } from '@assistant-ui/core/internal'
import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { useRuntimeMessageRepository } from './runtime-repository'

function assistant(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }], ...extra }
}

function user(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }], ...extra }
}

/**
 * Rebuild a live assistant-ui MessageRepository from the exported array the
 * hook produces. `import()` is what the incremental runtime does on every
 * transcript sync, and it links each node via `performOp` — which THROWS
 * "A message with the same id already exists in the parent tree" if two nodes
 * in one parent chain share an id. This is the exact crash Ace hit, so a real
 * gate has to exercise this path, not just count ids.
 */
function linkThroughRepository(exported: ReturnType<typeof useRuntimeMessageRepository>): MessageRepository {
  const repo = new MessageRepository()
  repo.import(exported)

  return repo
}

describe('useRuntimeMessageRepository — duplicate message id (performOp/link crash)', () => {
  it('does not crash when two messages resolve to the same adapter id', () => {
    // The adapter derives ids as `${timestamp}-${index}-${role}`. A resume /
    // rewind merge can splice two independently-converted arrays together, so
    // two DIFFERENT turns end up carrying the SAME id string. Pre-fix this
    // duplicate reached the repository and crashed the whole renderer.
    const messages: ChatMessage[] = [
      user('1784158733-0-user', 'first question'),
      assistant('1784158733-1-assistant', 'first answer'),
      // collision: same id string as the assistant above (merge artifact)
      assistant('1784158733-1-assistant', 'stale duplicate from resume-reconcile'),
      user('1784158740-3-user', 'second question'),
      assistant('1784158740-4-assistant', 'second answer')
    ]

    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    // The whole point: importing into a real MessageRepository must NOT throw.
    expect(() => linkThroughRepository(result.current)).not.toThrow()

    // And no message is dropped — every turn survives with a unique id.
    const ids = result.current.messages.map(({ message }) => message.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(result.current.messages).toHaveLength(messages.length)
  })

  it('preserves a hidden rewind branch that shares a sibling id', () => {
    const messages: ChatMessage[] = [
      user('1784200000-0-user', 'prompt'),
      assistant('1784200000-1-assistant', 'visible answer'),
      // hidden branch copy that collides with the visible sibling's id
      assistant('1784200000-1-assistant', 'hidden branch answer', {
        hidden: true,
        branchGroupId: 'grp-a'
      })
    ]

    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    expect(() => linkThroughRepository(result.current)).not.toThrow()

    const ids = result.current.messages.map(({ message }) => message.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('leaves already-unique ids untouched', () => {
    const messages: ChatMessage[] = [
      user('u-1', 'q'),
      assistant('a-1', 'answer one'),
      user('u-2', 'q2'),
      assistant('a-2', 'answer two')
    ]

    const { result } = renderHook(() => useRuntimeMessageRepository(messages))

    expect(() => linkThroughRepository(result.current)).not.toThrow()
    expect(result.current.messages.map(({ message }) => message.id)).toEqual(['u-1', 'a-1', 'u-2', 'a-2'])
  })
})
