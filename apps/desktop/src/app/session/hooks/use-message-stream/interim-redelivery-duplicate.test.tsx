import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

// A turn whose reply was sealed as an interim BEFORE the client had it on
// screen (the interim materialized its own bubble because a superseded
// attempt's frames had cleared the live stream) paints the reply twice on
// current main: the sealed interim above, and the settled bubble the same text
// streams into below — while the store holds one row (#123801).
//
// No user prompt row is mounted here on purpose: the duplicate is a
// render-side artifact of the interim/delta/complete ordering, not of the
// prompt row, and this keeps the fixture to the frames that matter. The
// boundary cases that must still paint twice have their own specs below.
const SID = 'interim-redelivery-duplicate'

const REPLY =
  'The Chinese posts all point at one third-party directory: it claims 1081 Meta Muse use cases across 28 pages, and the invite-code replies are the bulk of the chatter.'

const PROSE = 'Let me check that directory before I answer.'

let stream: MessageStreamHarness

const mountStream = () => {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
}

const start = () => act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

const delta = (text: string) => act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.delta' }))

const interim = (text: string) =>
  act(() => stream.handleEvent({ payload: { text, already_streamed: true }, session_id: SID, type: 'message.interim' }))

const tool = (name: string) =>
  act(() => {
    stream.handleEvent({ payload: { args: { command: 'ls' }, name, tool_id: 'call-1' }, session_id: SID, type: 'tool.start' })
    stream.handleEvent({ payload: { name, result: 'ok', tool_id: 'call-1' }, session_id: SID, type: 'tool.complete' })
  })

const complete = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.complete' }))

const flushDeltas = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS + 10)
  })

const state = (): ClientSessionState => stream.state()

/** Every visible assistant bubble's text, in transcript order. */
const visibleAssistantTexts = (): string[] =>
  state()
    .messages.filter(message => message.role === 'assistant' && !message.hidden)
    .map(message => chatMessageText(message).trim())
    .filter(Boolean)

const toolCallCount = (): number =>
  state().messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call').length

describe('a reply the turn re-streams after sealing it as an interim paints once', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('keeps one bubble when the sealed interim already carries the final reply', async () => {
    mountStream()
    await start()
    await interim(REPLY)
    await delta(REPLY)
    await flushDeltas()
    await complete(REPLY)

    expect(visibleAssistantTexts()).toEqual([REPLY])
  })

  it('never reaches across a message.start boundary to settle an earlier occurrence', async () => {
    mountStream()
    await start()
    await interim(REPLY)
    // A chained / prompt-less turn begins: message.start starts a new
    // occurrence and resets interimBoundaryPending while keeping the messages,
    // so the sealed interim above is now the PREVIOUS turn's segment. Matching
    // it would delete this turn's live bubble and complete the old one.
    await start()
    await delta(REPLY)
    await flushDeltas()
    await complete(REPLY)

    expect(visibleAssistantTexts()).toEqual([REPLY, REPLY])
  })

  it('still keeps a distinct interim segment beside the reply', async () => {
    mountStream()
    await start()
    await interim(PROSE)
    await delta(REPLY)
    await flushDeltas()
    await complete(REPLY)

    expect(visibleAssistantTexts()).toEqual([PROSE, REPLY])
  })

  it('never drops an interim that carries tool rows', async () => {
    mountStream()
    await start()
    await tool('terminal')
    await delta(REPLY)
    await flushDeltas()
    await interim(REPLY)
    await delta(REPLY)
    await flushDeltas()
    await complete(REPLY)

    // The interim owns the tool row, so it survives even though its text
    // matches the reply: dropping it would lose the completed tool card.
    expect(toolCallCount()).toBe(1)
    expect(visibleAssistantTexts()).toEqual([REPLY, REPLY])
  })
})
