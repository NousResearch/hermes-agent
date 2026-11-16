import type { GatewayEvent } from '@hermes/shared'
// A stopped mark says the reply ended there. A host plugin can pause a running
// turn (session.interrupt) to apply a grant and then continue the same reply
// with a hidden prompt: the stored read after the interrupted completion lends
// the stopped mark to the tool bubble, and the continuation streams on below
// it with no prompt on screen. That reply was not stopped for the reader. A
// prompt the user writes next keeps the mark, and so does a real Stop.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { finalizeStoppedMessages } from '@/app/session/hooks/use-prompt-actions/rewind'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'continued-reply-session'

const prompt: SessionMessage = { id: 1, role: 'user', content: 'can you pull example.com?', timestamp: 1 }

// What the store holds once the turn was stopped during its tool run: the call,
// its interrupted result, and the empty closing row that carries the mark.
const stoppedRows: SessionMessage[] = [
  prompt,
  {
    id: 2,
    role: 'assistant',
    content: '',
    timestamp: 2,
    tool_calls: [
      { id: 'call-1', type: 'function', function: { name: 'web_extract', arguments: '{"urls":["example.com"]}' } }
    ]
  },
  { id: 3, role: 'tool', content: '[Command interrupted]', tool_call_id: 'call-1', timestamp: 3 },
  {
    id: 4,
    role: 'assistant',
    content: '',
    display_kind: 'hidden',
    timestamp: 4,
    display_metadata: '{"interrupted": true}'
  }
]

const continuedRows: SessionMessage[] = [
  ...stoppedRows,
  { id: 5, role: 'user', content: 'Continue where you left off.', display_kind: 'hidden', timestamp: 5 },
  {
    id: 6,
    role: 'assistant',
    content: '',
    timestamp: 6,
    tool_calls: [
      { id: 'call-2', type: 'function', function: { name: 'web_extract', arguments: '{"urls":["example.com"]}' } }
    ]
  },
  { id: 7, role: 'tool', content: 'Example Domain', tool_call_id: 'call-2', timestamp: 7 },
  { id: 8, role: 'assistant', content: 'Here is example.com.', timestamp: 8 }
]

let stream: MessageStreamHarness

const emit = (event: Omit<GatewayEvent, 'session_id'>) => act(() => stream.handleEvent({ ...event, session_id: SID }))

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

// A refresh from stored history (the read after a turn, a reload) replaces the view.
const readStored = (rows: SessionMessage[]) =>
  act(() => {
    stream.states.set(SID, { ...stream.state(), messages: toChatMessages(rows) })
  })

// The mark the thread draws: AssistantMessage's StoppedNotice reads this flag.
const drawnStopped = (message: ChatMessage) => toRuntimeMessage(message).metadata?.custom?.interrupted === true

const replies = () => stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)

async function mountStoppedByPlugin() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  readStored([prompt])
  emit({ payload: {}, type: 'message.start' })
  emit({ payload: { args: { urls: ['example.com'] }, name: 'web_extract', tool_id: 'call-1' }, type: 'tool.start' })
  // The plugin's session.interrupt: the turn ends during its tool run.
  emit({ payload: { status: 'interrupted', text: '' }, type: 'message.complete' })
  readStored(stoppedRows)
}

describe('a reply that went on after a pause is not drawn as stopped', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('drops the mark once the hidden continuation starts, and keeps it off after the stored read', async () => {
    await mountStoppedByPlugin()
    expect(replies().map(drawnStopped)).toEqual([true])

    emit({ payload: {}, type: 'message.start' })
    expect(replies().map(drawnStopped)).toEqual([false])
    // Display only: the stopped turn's own record still says it was stopped.
    expect(replies()[0].interrupted).toBe(true)

    emit({ payload: { args: { urls: ['example.com'] }, name: 'web_extract', tool_id: 'call-2' }, type: 'tool.start' })
    emit({ payload: { name: 'web_extract', result: 'Example Domain', tool_id: 'call-2' }, type: 'tool.complete' })
    emit({ payload: { text: 'Here is example.com.' }, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { status: 'complete', text: 'Here is example.com.' }, type: 'message.complete' })
    expect(replies().some(drawnStopped)).toBe(false)

    readStored(continuedRows)
    expect(replies().some(drawnStopped)).toBe(false)
  })

  it('keeps the mark when the user writes next', async () => {
    await mountStoppedByPlugin()
    act(() => {
      const state = stream.state()
      stream.states.set(SID, {
        ...state,
        messages: [...state.messages, { id: 'user-next', role: 'user', parts: [{ type: 'text', text: 'never mind' }] }]
      })
    })

    emit({ payload: {}, type: 'message.start' })

    expect(replies().map(drawnStopped)).toEqual([true])
  })

  it('keeps the mark when an automation card starts the next turn', async () => {
    await mountStoppedByPlugin()

    emit({ payload: { display_kind: 'process_complete', display_text: 'build finished' }, type: 'message.start' })

    expect(replies().map(drawnStopped)).toEqual([true])
  })

  it('keeps the mark of a real Stop, live and after the stored read', async () => {
    vi.useFakeTimers()
    stream = renderMessageStream(SID)
    readStored([prompt])
    emit({ payload: {}, type: 'message.start' })
    emit({ payload: { args: { urls: ['example.com'] }, name: 'web_extract', tool_id: 'call-1' }, type: 'tool.start' })
    act(() => {
      const state = stream.state()
      stream.states.set(SID, {
        ...state,
        messages: finalizeStoppedMessages(state.messages, state.streamId),
        busy: false,
        awaitingResponse: false,
        streamId: null,
        interrupted: true,
        turnLive: false
      })
    })
    emit({ payload: { status: 'interrupted', text: '' }, type: 'message.complete' })
    expect(replies().map(drawnStopped)).toEqual([true])

    readStored(stoppedRows)
    expect(replies().map(drawnStopped)).toEqual([true])
  })
})
