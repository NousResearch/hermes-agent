import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest, hasClarifyRequest } from '@/store/clarify'
import { onScrollToBottomRequest } from '@/store/thread-scroll'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// A `clarify.request` must leave an answerable inline row even when the
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

const clarifyRequest = (payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'clarify.request' }))

const toolStart = (payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'tool.start' }))

const toolComplete = (payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'tool.complete' }))

const clarifyExpire = (requestId: string) =>
  act(() => stream.handleEvent({ payload: { request_id: requestId }, session_id: SID, type: 'clarify.expire' }))

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

describe('clarify.request stream hydration', () => {
  beforeEach(() => {
    clearClarifyRequest()
    scrollToBottom.mockClear()
    stopScrollListener = onScrollToBottomRequest(scrollToBottom, SID)
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    stopScrollListener?.()
    stopScrollListener = null
    vi.restoreAllMocks()
  })

  it('mounts an answerable clarify row when the tool.start row was missed', () => {
    mountStream()

    clarifyRequest({ choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-1' })

    const parts = clarifyParts()
    expect(parts).toHaveLength(1)
    expect(parts[0].type === 'tool-call' && parts[0].toolCallId).toBe('req-1')
    expect(parts[0].type === 'tool-call' && parts[0].args).toMatchObject({
      choices: ['yes', 'no'],
      question: 'Ship it?'
    })
  })

  it('reveals a clarify prompt raised by the active session', () => {
    mountStream()

    clarifyRequest({ choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-reveal' })

    expect(scrollToBottom).toHaveBeenCalledOnce()
  })

  it('does not move the active thread for a background session clarify', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-background' },
        session_id: 'session-background',
        type: 'clarify.request'
      })
    )

    expect(scrollToBottom).not.toHaveBeenCalled()
  })

  it('preserves multi-select through the store and hydrated tool row', () => {
    mountStream()

    clarifyRequest({
      choices: ['read', 'write'],
      multi_select: true,
      question: 'Which permissions?',
      request_id: 'req-multi'
    })

    expect($clarifyRequests.get()[SID]?.multiSelect).toBe(true)

    const part = clarifyParts()[0]
    expect(part?.type).toBe('tool-call')

    if (part?.type !== 'tool-call') {
      throw new Error('Expected a hydrated clarify tool call')
    }

    expect(part.args).toMatchObject({
      choices: ['read', 'write'],
      multi_select: true,
      question: 'Which permissions?'
    })
  })

  it('merges with the real tool.start row even though its id differs from the request id', () => {
    mountStream()

    // Reality: tool.start carries the model's tool_call_id, clarify.request a
    // separately-generated request_id. They must still collapse to ONE card
    // (correlated by question), not two.
    toolStart({ args: { choices: ['a'], question: 'Pick' }, name: 'clarify', tool_id: 'call-abc' })
    clarifyRequest({ choices: ['a'], question: 'Pick', request_id: 'req-2' })

    expect(clarifyParts()).toHaveLength(1)
  })

  it('does not duplicate when clarify.request arrives before the tool.start row', () => {
    mountStream()

    clarifyRequest({ choices: ['a'], question: 'Pick', request_id: 'req-3' })
    toolStart({ args: { choices: ['a'], question: 'Pick' }, name: 'clarify', tool_id: 'call-xyz' })

    expect(clarifyParts()).toHaveLength(1)
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
            args: { choices: ['a', 'b'], question: 'Pick' },
            argsText: '{"question":"Pick","choices":["a","b"]}'
          }
        ]
      }
    ])

    clarifyRequest({ choices: ['a', 'b'], question: 'Pick', request_id: 'req-codex' })

    const messages = stream.state().messages
    expect(messages).toHaveLength(2)
    expect(clarifyParts()).toHaveLength(1)
    expect(messages[1]).toMatchObject({ id: 'assistant-codex', pending: true })
    expect(stream.state().streamId).toBe('assistant-codex')
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
            args: { choices: ['safe', 'fast'], question: 'Which path?' },
            argsText: '{"question":"Which path?","choices":["safe","fast"]}'
          }
        ]
      }
    ])

    clarifyRequest({ choices: ['safe', 'fast'], question: 'Which path?', request_id: 'req-deepseek' })

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
            args: { choices: ['safe', 'fast'], question: 'Which path?' },
            argsText: '{"question":"Which path?","choices":["safe","fast"]}'
          }
        ]
      }
    ])

    clarifyRequest({ choices: ['safe', 'fast'], question: 'Which path?', request_id: 'req-ui' })
    toolComplete({
      args: { choices: ['safe', 'fast'], question: 'Which path?' },
      name: 'clarify',
      result: { question: 'Which path?', user_response: 'safe' },
      tool_id: 'call-provider'
    })

    const parts = clarifyParts()
    expect(parts).toHaveLength(1)
    expect(parts[0]).toMatchObject({ toolCallId: 'call-provider', result: { user_response: 'safe' } })
  })

  it('ignores a late clarify.request after the turn was interrupted', () => {
    mountStream()
    seedHydratedMessages([{ id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'stop this' }] }])

    const state = stream.states.get(SID)!
    state.interrupted = true

    clarifyRequest({ choices: ['a', 'b'], question: 'Pick', request_id: 'req-late' })

    expect($clarifyRequests.get()[SID]).toBeUndefined()
    expect(stream.state().messages).toHaveLength(1)
  })

  it('expires only the matching clarify request and deactivates its card', () => {
    mountStream()

    toolStart({ args: { choices: ['a'], question: 'Pick' }, name: 'clarify', tool_id: 'call-provider' })
    clarifyRequest({ choices: ['a'], question: 'Pick', request_id: 'req-expire' })
    clarifyExpire('req-other')

    expect($clarifyRequests.get()[SID]?.requestId).toBe('req-expire')
    expect(clarifyParts()[0]).not.toHaveProperty('result')

    clarifyExpire('req-expire')

    expect($clarifyRequests.get()[SID]).toBeUndefined()
    expect(clarifyParts()).toHaveLength(1)
    expect(clarifyParts()[0]).toHaveProperty('result')
    expect(stream.state().needsInput).toBe(false)
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

const REMOTE_SID = 'runtime-remote-1'
const REMOTE_QUESTION = 'Authorize the exact sandbox credential-read boundary?'

const remoteEvent = (type: string, payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: REMOTE_SID, type }))

const startClarify = (toolId: string) =>
  remoteEvent('tool.start', {
    args: { choices: ['Allow', 'Deny'], question: REMOTE_QUESTION },
    name: 'clarify',
    tool_id: toolId
  })

const requestClarify = (requestId: string) =>
  remoteEvent('clarify.request', { choices: ['Allow', 'Deny'], question: REMOTE_QUESTION, request_id: requestId })

const completeClarify = (toolId: string) =>
  remoteEvent('tool.complete', {
    args: { question: REMOTE_QUESTION },
    name: 'clarify',
    result: JSON.stringify({ question: REMOTE_QUESTION, user_response: 'Allow' }),
    tool_id: toolId
  })

describe('clarify answered by another renderer', () => {
  beforeEach(() => {
    clearClarifyRequest()
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
  })

  it('settles the gateway request when the completion carries only the model tool-call id', () => {
    const states = new Map()
    stream = renderMessageStream(REMOTE_SID, { states })

    startClarify('call-remote')
    requestClarify('req-remote')

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-remote')
    expect(states.get(REMOTE_SID)?.needsInput).toBe(true)

    completeClarify('call-remote')

    expect($clarifyRequests.get()[REMOTE_SID]).toBeUndefined()
    expect(hasClarifyRequest(REMOTE_SID)).toBe(false)
    expect(states.get(REMOTE_SID)?.needsInput).toBe(false)

    const unresolved = (stream.state(REMOTE_SID).messages ?? [])
      .flatMap(message => message.parts)
      .filter(part => part.type === 'tool-call' && part.toolName === 'clarify' && part.result === undefined)

    expect(unresolved).toHaveLength(0)
  })

  it('leaves the live request alone when an OLDER epoch’s clarify completes late', () => {
    const states = new Map()
    stream = renderMessageStream(REMOTE_SID, { states })

    startClarify('call-epoch-1')
    requestClarify('req-epoch-1')
    startClarify('call-epoch-2')
    requestClarify('req-epoch-2')

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-epoch-2')

    completeClarify('call-epoch-1')

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-epoch-2')
    expect(states.get(REMOTE_SID)?.needsInput).toBe(true)
  })

  it('still settles when an unrelated tool completes in between', () => {
    const states = new Map()
    stream = renderMessageStream(REMOTE_SID, { states })

    startClarify('call-mixed')
    requestClarify('req-mixed')

    remoteEvent('tool.start', { args: { path: 'notes.md' }, name: 'read_file', tool_id: 'call-read-9' })
    remoteEvent('tool.complete', { name: 'read_file', result: 'ok', tool_id: 'call-read-9' })

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-mixed')
    expect(states.get(REMOTE_SID)?.needsInput).toBe(true)

    completeClarify('call-mixed')

    expect(hasClarifyRequest(REMOTE_SID)).toBe(false)
    expect(states.get(REMOTE_SID)?.needsInput).toBe(false)
  })

  it('does not settle or clear attention on a clarify completion with no identity and no question', () => {
    const states = new Map()
    stream = renderMessageStream(REMOTE_SID, { states })

    startClarify('call-malformed')
    requestClarify('req-malformed')

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-malformed')
    expect(states.get(REMOTE_SID)?.needsInput).toBe(true)

    remoteEvent('tool.complete', { name: 'clarify', result: 'ok' })

    expect($clarifyRequests.get()[REMOTE_SID]?.requestId).toBe('req-malformed')
    expect(hasClarifyRequest(REMOTE_SID)).toBe(true)
    expect(states.get(REMOTE_SID)?.needsInput).toBe(true)
  })
})

const LONG_SID = 'runtime-long-1'
const FIELD_QUESTION = 'Authorize the exact sandbox credential-read boundary?'

function longPersistedHistory() {
  const messages = []

  for (let index = 0; index < 740; index += 1) {
    messages.push({
      id: `row-${index}`,
      parts: [{ type: 'text' as const, text: `persisted turn ${index}` }],
      role: index % 2 === 0 ? ('user' as const) : ('assistant' as const),
      rowId: index
    })
  }

  messages.push({
    id: 'inflight-assistant-segment-0-runtime-long-1',
    interim: true,
    parts: [{ type: 'text' as const, text: 'sealed interim boundary' }],
    pending: false,
    role: 'assistant' as const
  })

  return messages
}

function unresolvedClarifyRows() {
  return (stream.state(LONG_SID).messages ?? [])
    .flatMap(m => m.parts)
    .filter(p => p.type === 'tool-call' && p.toolName === 'clarify' && p.result === undefined)
}

describe('durable clarify projection across a long session', () => {
  beforeEach(() => {
    clearClarifyRequest()
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
  })

  it('keeps clarify attention when an unrelated tool completes while the request is unresolved', () => {
    const states = new Map()
    stream = renderMessageStream(LONG_SID, { states })

    act(() =>
      stream.handleEvent({
        payload: { messages: longPersistedHistory() },
        session_id: LONG_SID,
        type: 'session.info'
      })
    )

    act(() =>
      stream.handleEvent({
        payload: { args: { path: 'notes.md' }, name: 'read_file', tool_id: 'call-read-1' },
        session_id: LONG_SID,
        type: 'tool.start'
      })
    )

    act(() =>
      stream.handleEvent({
        payload: {
          choices: ['Allow', 'Deny'],
          question: FIELD_QUESTION,
          request_id: 'req-field-1'
        },
        session_id: LONG_SID,
        type: 'clarify.request'
      })
    )

    expect($clarifyRequests.get()[LONG_SID]?.requestId).toBe('req-field-1')
    expect(unresolvedClarifyRows()).toHaveLength(1)
    expect(states.get(LONG_SID)?.needsInput).toBe(true)

    act(() =>
      stream.handleEvent({
        payload: { name: 'read_file', result: 'ok', tool_id: 'call-read-1' },
        session_id: LONG_SID,
        type: 'tool.complete'
      })
    )

    expect($clarifyRequests.get()[LONG_SID]?.requestId).toBe('req-field-1')
    expect(unresolvedClarifyRows()).toHaveLength(1)
    expect(states.get(LONG_SID)?.needsInput).toBe(true)
  })

  it('clears clarify attention exactly once on the matching clarify completion', () => {
    const states = new Map()
    stream = renderMessageStream(LONG_SID, { states })

    act(() =>
      stream.handleEvent({
        payload: { choices: ['Allow', 'Deny'], question: FIELD_QUESTION, request_id: 'req-field-2' },
        session_id: LONG_SID,
        type: 'clarify.request'
      })
    )

    expect(states.get(LONG_SID)?.needsInput).toBe(true)

    act(() =>
      stream.handleEvent({
        payload: {
          args: { question: FIELD_QUESTION },
          name: 'clarify',
          result: JSON.stringify({ question: FIELD_QUESTION, user_response: 'Allow' }),
          tool_id: 'req-field-2'
        },
        session_id: LONG_SID,
        type: 'tool.complete'
      })
    )

    expect(states.get(LONG_SID)?.needsInput).toBe(false)
    expect($clarifyRequests.get()[LONG_SID]).toBeUndefined()
  })
})
