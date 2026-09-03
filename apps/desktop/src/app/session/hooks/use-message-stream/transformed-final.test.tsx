import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { clearSessionTodos } from '@/store/todos'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

const start = () => act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

const delta = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.delta' }))

const interim = (text: string) =>
  act(() => stream.handleEvent({ payload: { text, already_streamed: true }, session_id: SID, type: 'message.interim' }))

const completeTransformed = (text: string) =>
  act(() =>
    stream.handleEvent({ payload: { text, response_transformed: true }, session_id: SID, type: 'message.complete' })
  )

function getState(): ClientSessionState {
  return stream.state()
}

function assistantTexts(): string[] {
  const state = getState()
  return state.messages
    .filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
    .filter(Boolean)
}

describe('useMessageStream response_transformed settlement', () => {
  beforeEach(() => {
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('replaces the streamed bubble with transformed final text that shares no prefix', async () => {
    mountStream()
    await start()

    // A transform_llm_output plugin hook (e.g. pseudonym restore) rewrites the
    // final text after streaming finishes. The rewritten text shares NO prefix
    // relationship with what was streamed: the prefix-continuity heuristic must
    // not reject it — it is this turn's authoritative reply.
    await delta('TOKEN_1')
    await interim('TOKEN_1')
    await completeTransformed('example-service.internal')

    const texts = assistantTexts()
    expect(texts).toHaveLength(1)
    expect(texts[0]).toBe('example-service.internal')
    expect(texts).not.toContain('TOKEN_1')
  })

  it('clears the interim mark when the transformed final settles onto the bubble', async () => {
    mountStream()
    await start()

    await delta('TOKEN_1')
    await interim('TOKEN_1')
    await completeTransformed('example-service.internal')

    const assistants = getState().messages.filter(m => m.role === 'assistant' && !m.hidden)
    expect(assistants).toHaveLength(1)
    expect(assistants[0].interim).toBeFalsy()
    expect(assistants[0].pending).toBe(false)
  })

  it('does not overwrite an unrelated bubble after a message.start reset', async () => {
    mountStream()
    await start()
    await interim('old reply')

    // A new turn begins: message.start resets sawAssistantPayload. A stale or
    // foreign transformed completion must NOT overwrite the previous bubble —
    // the sawAssistantPayload gate keeps it from settling there.
    await start()
    await completeTransformed('totally new answer')

    const texts = assistantTexts()
    expect(texts).toContain('old reply')
    expect(texts).toContain('totally new answer')
    expect(texts).toHaveLength(2)
  })
})
