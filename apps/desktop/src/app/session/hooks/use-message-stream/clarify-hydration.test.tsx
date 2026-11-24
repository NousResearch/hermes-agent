import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { onScrollToBottomRequest } from '@/store/thread-scroll'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// A `clarify` server request must leave an answerable inline row even when the
// `tool.start` that normally mounts it was missed (stream reconnect /
// hydration race). Without it the sidebar says "needs input" but the
// transcript has nowhere to render the choices, so the agent blocks forever.

const SID = 'session-1'

let stream: MessageStreamHarness
let stopScrollListener: (() => void) | null = null

const scrollToBottom = vi.fn()

function mountStream() {
  stream = renderMessageStream(SID)
}

const clarifyRequest = ({ request_id, ...params }: Record<string, unknown>) =>
  act(() => void stream.handleRequest('clarify', { ...params, session_id: SID }, request_id as string))

const toolStart = (payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'tool.start' }))

const toolComplete = (payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'tool.complete' }))

const clarifyExpire = (requestId: string) =>
  act(() =>
    stream.handleEvent({
      payload: { id: requestId, method: 'clarify', reason: 'timeout' },
      session_id: SID,
      type: 'request.cancel'
    })
  )

function clarifyParts() {
  const messages = stream.state().messages ?? []

  return messages.flatMap(m => m.parts).filter(p => p.type === 'tool-call' && p.toolName === 'clarify')
}

function seedHydratedMessages(messages: ChatMessage[]) {
  const state = createClientSessionState()
  state.messages = messages
  state.streamId = null
  stream.states.set(SID, state)
}

describe('clarify request stream hydration', () => {
  beforeEach(() => {
    clearClarifyRequest()
    resetServerRequestsForTests()
    scrollToBottom.mockClear()
    stopScrollListener = onScrollToBottomRequest(scrollToBottom, SID)
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    resetServerRequestsForTests()
    stopScrollListener?.()
    stopScrollListener = null
    vi.restoreAllMocks()
  })

  it('mounts an answerable clarify row when the tool.start row was missed', () => {
    mountStream()

    clarifyRequest({ questions: [{ choices: ['yes', 'no'], qid: 'q0', question: 'Ship it?' }], request_id: 'req-1' })

    const parts = clarifyParts()
    expect(parts).toHaveLength(1)
    expect(parts[0].type === 'tool-call' && parts[0].toolCallId).toBe('req-1')
    expect(parts[0].type === 'tool-call' && parts[0].args).toMatchObject({
      questions: [{ choices: ['yes', 'no'], question: 'Ship it?' }]
    })
  })

  it('reveals a clarify prompt raised by the active session', () => {
    mountStream()

    clarifyRequest({
      questions: [{ choices: ['yes', 'no'], qid: 'q0', question: 'Ship it?' }],
      request_id: 'req-reveal'
    })

    expect(scrollToBottom).toHaveBeenCalledOnce()
  })

  it('does not move the active thread for a background session clarify', () => {
    mountStream()

    act(
      () =>
        void stream.handleRequest(
          'clarify',
          {
            questions: [{ choices: ['yes', 'no'], qid: 'q0', question: 'Ship it?' }],
            session_id: 'session-background'
          },
          'req-background'
        )
    )

    expect(scrollToBottom).not.toHaveBeenCalled()
  })

  it('preserves multi-select through the store and hydrated tool row', () => {
    mountStream()

    clarifyRequest({
      questions: [{ choices: ['read', 'write'], multi_select: true, qid: 'q0', question: 'Which permissions?' }],
      request_id: 'req-multi'
    })

    expect($clarifyRequests.get()[SID]?.questions[0]?.multiSelect).toBe(true)

    const part = clarifyParts()[0]
    expect(part?.type).toBe('tool-call')

    if (part?.type !== 'tool-call') {
      throw new Error('Expected a hydrated clarify tool call')
    }

    expect(part.args).toMatchObject({
      questions: [{ choices: ['read', 'write'], multi_select: true, question: 'Which permissions?' }]
    })
  })

  it('re-arms a hydrated Codex tool-only clarify in place instead of appending a second card', () => {
    mountStream()

    seedHydratedMessages([
      { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'help me choose' }] },
      {
        id: 'assistant-codex',
        role: 'assistant',
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'call-codex',
            toolName: 'clarify',
            args: { questions: [{ choices: ['a', 'b'], question: 'Pick' }] },
            argsText: '{"questions":[{"question":"Pick","choices":["a","b"]}]}'
          }
        ]
      }
    ])

    clarifyRequest({ questions: [{ choices: ['a', 'b'], qid: 'q0', question: 'Pick' }], request_id: 'req-codex' })

    const messages = stream.state().messages
    expect(messages).toHaveLength(2)
    expect(clarifyParts()).toHaveLength(1)
    expect(messages[1]).toMatchObject({ id: 'assistant-codex', pending: true })
    expect(stream.state().streamId).toBe('assistant-codex')
  })

  it('hydrates an already-pending sparse clarify row from the live request', () => {
    mountStream()

    seedHydratedMessages([
      { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'help me choose' }] },
      {
        id: 'assistant-pending',
        role: 'assistant',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'call-provider',
            toolName: 'clarify',
            args: {},
            argsText: '{}'
          }
        ]
      }
    ])

    clarifyRequest({
      questions: [{ choices: ['Approve', 'Reject'], qid: 'q0', question: 'Continue?' }],
      request_id: 'req-pending'
    })

    const messages = stream.state().messages
    const part = clarifyParts()[0]
    expect(messages).toHaveLength(2)
    expect(messages[1]).toMatchObject({ id: 'assistant-pending', pending: true })
    expect(part).toMatchObject({
      toolCallId: 'call-provider',
      args: { questions: [{ choices: ['Approve', 'Reject'], question: 'Continue?' }] }
    })
    expect(stream.state().streamId).toBe('assistant-pending')
  })

  it('keeps a hydrated DeepSeek text-plus-clarify row in its original position', () => {
    mountStream()

    seedHydratedMessages([
      { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'inspect this' }] },
      {
        id: 'assistant-deepseek',
        role: 'assistant',
        parts: [
          { type: 'text', text: 'I found two paths; choose one.' },
          {
            type: 'tool-call',
            toolCallId: 'call-deepseek',
            toolName: 'clarify',
            args: { questions: [{ choices: ['safe', 'fast'], question: 'Which path?' }] },
            argsText: '{"questions":[{"question":"Which path?","choices":["safe","fast"]}]}'
          }
        ]
      }
    ])

    clarifyRequest({
      questions: [{ choices: ['safe', 'fast'], qid: 'q0', question: 'Which path?' }],
      request_id: 'req-deepseek'
    })

    const messages = stream.state().messages
    expect(messages).toHaveLength(2)
    expect(messages[1].id).toBe('assistant-deepseek')
    expect(messages[1].parts.map(part => part.type)).toEqual(['text', 'tool-call'])
    expect(messages[1].pending).toBe(true)
    expect(stream.state().streamId).toBe('assistant-deepseek')
  })

  it('settles the re-armed provider tool id in place when tool.complete arrives', () => {
    mountStream()

    seedHydratedMessages([
      { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'inspect this' }] },
      {
        id: 'assistant-deepseek',
        role: 'assistant',
        parts: [
          { type: 'text', text: 'I found two paths; choose one.' },
          {
            type: 'tool-call',
            toolCallId: 'call-provider',
            toolName: 'clarify',
            args: { questions: [{ choices: ['safe', 'fast'], question: 'Which path?' }] },
            argsText: '{"questions":[{"question":"Which path?","choices":["safe","fast"]}]}'
          }
        ]
      }
    ])

    clarifyRequest({
      questions: [{ choices: ['safe', 'fast'], qid: 'q0', question: 'Which path?' }],
      request_id: 'req-ui'
    })
    toolComplete({
      args: { questions: [{ choices: ['safe', 'fast'], question: 'Which path?' }] },
      name: 'clarify',
      result: {
        outcome: 'submitted',
        responses: [
          { choices_offered: ['safe', 'fast'], question: 'Which path?', status: 'answered', user_response: 'safe' }
        ]
      },
      tool_id: 'call-provider'
    })

    const parts = clarifyParts()
    expect(parts).toHaveLength(1)
    expect(parts[0]).toMatchObject({ toolCallId: 'call-provider', result: { outcome: 'submitted' } })
  })

  it('ignores a late clarify.request after the turn was interrupted', () => {
    mountStream()
    seedHydratedMessages([{ id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'stop this' }] }])

    const state = stream.states.get(SID)!
    state.interrupted = true

    clarifyRequest({ questions: [{ choices: ['a', 'b'], qid: 'q0', question: 'Pick' }], request_id: 'req-late' })

    expect($clarifyRequests.get()[SID]).toBeUndefined()
    expect(stream.state().messages).toHaveLength(1)
  })

  it('expires only the matching clarify request and deactivates its card', () => {
    mountStream()

    toolStart({
      args: { questions: [{ choices: ['a'], question: 'Pick' }] },
      name: 'clarify',
      tool_id: 'call-provider'
    })
    clarifyRequest({ questions: [{ choices: ['a'], qid: 'q0', question: 'Pick' }], request_id: 'req-expire' })
    clarifyExpire('req-other')

    expect($clarifyRequests.get()[SID]?.requestId).toBe('req-expire')
    expect(clarifyParts()[0]).not.toHaveProperty('result')

    clarifyExpire('req-expire')

    expect($clarifyRequests.get()[SID]).toBeUndefined()
    expect(clarifyParts()).toHaveLength(1)
    expect(clarifyParts()[0]).toHaveProperty('result')
    expect(stream.state().needsInput).toBe(false)
  })

  it.each(['message.complete', 'error'] as const)('keeps a live clarify card through a spurious %s', type => {
    mountStream()
    clarifyRequest({ questions: [{ choices: ['a', 'b'], qid: 'q0', question: 'Pick' }], request_id: 'req-live' })

    act(() =>
      stream.handleEvent({
        payload: type === 'error' ? { message: 'spurious error' } : { text: '' },
        session_id: SID,
        type
      })
    )

    expect($clarifyRequests.get()[SID]?.requestId).toBe('req-live')

    clarifyExpire('req-live')
    expect($clarifyRequests.get()[SID]).toBeUndefined()
  })

  it('keeps a live clarify card through a stale session.info running=false snapshot (#83319)', () => {
    mountStream()

    // The state a reconnect replays from: the turn was live (busy) when the
    // backend parked the clarify request. The session.info snapshot riding
    // the reconnect predates the clarify, so its running=false is stale —
    // the same wipe class as the spurious turn-end/error clears.
    const state = createClientSessionState()
    state.busy = true
    state.awaitingResponse = true
    stream.states.set(SID, state)

    clarifyRequest({ questions: [{ choices: ['a', 'b'], qid: 'q0', question: 'Pick' }], request_id: 'req-live' })

    act(() => stream.handleEvent({ payload: { running: false }, session_id: SID, type: 'session.info' }))

    expect($clarifyRequests.get()[SID]?.requestId).toBe('req-live')

    // And the clear still fires once the request truly settles.
    clarifyExpire('req-live')
    expect($clarifyRequests.get()[SID]).toBeUndefined()
  })

  it('merges a BATCH tool.start row with its clarify.request (no top-level question)', () => {
    mountStream()

    // The batch shape: tool args carry `questions`, no top-level `question`.
    // The correlation key must come from the question list, or the two ids
    // mount two cards (the duplicate seen in the field).
    toolStart({
      args: { questions: [{ question: 'Drink?' }, { question: 'Productive when?' }] },
      name: 'clarify',
      tool_id: 'call-batch'
    })
    clarifyRequest({
      questions: [
        { qid: 'q0', question: 'Drink?' },
        { qid: 'q1', question: 'Productive when?' }
      ],
      request_id: 'req-batch'
    })

    expect(clarifyParts()).toHaveLength(1)
    expect($clarifyRequests.get()[SID]?.questions).toHaveLength(2)
  })

  it('does not duplicate when the batch clarify.request arrives before tool.start', () => {
    mountStream()

    clarifyRequest({
      questions: [
        { qid: 'q0', question: 'Drink?' },
        { qid: 'q1', question: 'Productive when?' }
      ],
      request_id: 'req-batch-2'
    })
    toolStart({
      args: { questions: [{ question: 'Drink?' }, { question: 'Productive when?' }] },
      name: 'clarify',
      tool_id: 'call-batch-2'
    })

    expect(clarifyParts()).toHaveLength(1)
    expect($clarifyRequests.get()[SID]?.questions).toHaveLength(2)
  })
})
