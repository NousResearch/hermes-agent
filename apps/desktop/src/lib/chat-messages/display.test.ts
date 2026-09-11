import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { toRuntimeMessage } from '@/lib/chat-runtime'

import { toChatMessages } from './hydration'
import { textPart } from './parts'
import type { ChatMessage } from './types'

afterEach(cleanup)

describe('silent response presentation invariants', () => {
  it('recognizes split control text while retaining the reasoning evidence', () => {
    const reasoning = { type: 'reasoning' as const, text: 'Internal checks completed' }
    const message: ChatMessage = { id: 'split', role: 'assistant', parts: [textPart('[[SI'), reasoning, textPart('LENT]]')] }
    expect(toRuntimeMessage(message).content).toEqual([reasoning])
    expect(message.parts).toHaveLength(3)
  })

  it('filters an exact final segment without losing the stored parent commentary', () => {
    const messages = toChatMessages([
      { role: 'user', content: 'Check the result.', timestamp: 1 },
      { role: 'assistant', content: 'The checks are complete.', timestamp: 2 },
      { role: 'assistant', content: '[[SILENT]]', timestamp: 3 },
    ])
    const raw = { ...messages.at(-1)!, parts: messages.filter(message => message.role === 'assistant').flatMap(message => message.parts) }
    const original = JSON.stringify(raw)
    const runtime = toRuntimeMessage(raw)!
    expect(runtime.content).toEqual([expect.objectContaining({ type: 'text', text: 'The checks are complete.' })])
    expect(JSON.stringify(raw)).toBe(original)
  })

  it('retains tool evidence while removing only the control text', () => {
    const message: ChatMessage = {
      id: 'tool-result', role: 'assistant', parts: [
        { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', argsText: '{}', result: 'verified' },
        textPart('[[SILENT]]'),
      ],
    }
    const runtime = toRuntimeMessage(message)!
    expect(runtime.content).toEqual([expect.objectContaining({ type: 'tool-call', toolCallId: 'call-1', result: 'verified' })])
    expect(runtime.metadata?.custom?.silent).not.toBe(true)
  })

  it('keeps silent turns in the repository lineage instead of merging adjacent answers', () => {
    const raw = toChatMessages([
      { role: 'user', content: 'First question', timestamp: 1 },
      { role: 'assistant', content: 'First answer', timestamp: 2 },
      { role: 'user', content: 'Second question', timestamp: 3 },
      { role: 'assistant', content: '[[SILENT]]', timestamp: 4 },
    ])
    const { result } = renderHook(() => useRuntimeMessageRepository(raw))
    const repository = result.current
    expect(repository.messages).toHaveLength(4)
    expect(repository.headId).toBe(raw[3].id)
    expect(repository.messages[3]).toMatchObject({
      parentId: raw[2].id,
      message: { id: raw[3].id, metadata: { custom: { silent: true } } },
    })
  })

  it('does not classify incomplete or malformed provenance as a delegation', () => {
    for (const display_metadata of ['invalid json', '{}', '{"user_originated":false}', '{"event_kind":"workflow.async_delegation.terminal"}']) {
      const [message] = toChatMessages([{ role: 'user', content: 'Keep this', display_kind: 'internal_event', display_metadata }])
      expect(message.role).toBe('user')
      expect(message.asyncResult).toBeUndefined()
    }
  })
})
