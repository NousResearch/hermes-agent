import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'late-tool-complete-session'

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

const toolParts = () =>
  (stream.state(SID).messages ?? []).flatMap(message => message.parts.filter(part => part.type === 'tool-call'))

describe('late tool completions across interim boundaries', () => {
  beforeEach(async () => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('attaches a late completion to the sealed bubble that owns the call', () => {
    event('message.start', 100)
    event('message.delta', 101, { text: 'Scraping the page.' })
    event('tool.start', 102, { args: { url: 'https://example.test' }, name: 'browser', tool_id: 'call-late-1' })
    // Interim commentary seals the streaming bubble while the browser scrape
    // keeps running (minutes in practice). streamId drops; the tool part is
    // sealed with completedAt and no result.
    event('message.interim', 103, { text: 'Scraping the page.' })
    event('tool.complete', 260.5, { name: 'browser', result: 'ok', tool_id: 'call-late-1' })

    const assistants = (stream.state(SID).messages ?? []).filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(1)
    // The sealed part got the result instead of staying "Result unavailable"...
    expect(assistants[0].parts).toEqual([
      expect.objectContaining({ completedAt: 102, type: 'text' }),
      expect.objectContaining({ result: 'ok', toolCallId: 'call-late-1', type: 'tool-call' })
    ])

    // ...and no duplicate row was seeded into a fresh bubble.
    expect(toolParts()).toHaveLength(1)
  })

  it('keeps the sealed bubble sealed after reconciling', () => {
    event('message.start', 200)
    event('tool.start', 201, { args: {}, name: 'browser', tool_id: 'call-late-2' })
    event('message.interim', 202, { text: 'Working on it.' })
    event('tool.complete', 203, { name: 'browser', result: 'ok', tool_id: 'call-late-2' })

    // The reconciliation must not reopen the stream onto the sealed bubble:
    // a later turn still seeds the next bubble instead of appending into it.
    event('message.delta', 204, { text: 'Follow-up text.' })
    event('message.complete', 205, { text: 'Follow-up text.' })

    const assistants = (stream.state(SID).messages ?? []).filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(2)
    expect(assistants[0]).toMatchObject({ completedAt: 202, interim: true })
    expect(
      assistants[0].parts.some(part => part.type === 'tool-call' && 'result' in part && part.result === 'ok')
    ).toBe(true)
    expect(assistants[1]).toMatchObject({ completedAt: 205 })
    expect(assistants[1].parts.map(part => part.type)).toEqual(['text'])
  })

  it('reconciles into the owning bubble even while a new bubble streams', () => {
    event('message.start', 300)
    event('tool.start', 301, { args: {}, name: 'browser', tool_id: 'call-late-3' })
    event('message.interim', 302, { text: 'Still scraping.' })
    // The turn continues after the boundary: a new bubble opens and streams.
    event('message.delta', 303, { text: 'Meanwhile.' })
    // The long tool finishes — its completion must land in the FIRST bubble.
    event('tool.complete', 304, { name: 'browser', result: 'done', tool_id: 'call-late-3' })

    const assistants = (stream.state(SID).messages ?? []).filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(2)
    expect(assistants[0].parts.map(part => part.type)).toEqual(['tool-call', 'text'])
    expect(assistants[0].parts[0]).toMatchObject({ result: 'done', toolCallId: 'call-late-3', type: 'tool-call' })
    // The streaming bubble keeps only its text — no duplicate tool row.
    expect(assistants[1].parts.map(part => part.type)).toEqual(['text'])
  })

  it('seeds a fresh bubble for a completion with no owner', () => {
    event('message.start', 400)
    event('message.interim', 401, { text: 'Sealed.' })
    // A completion whose tool.start never arrived (reconnect gap): no part
    // owns the id, so it seeds a bubble as before instead of vanishing.
    event('tool.complete', 402, { name: 'grep', result: 'match', tool_id: 'call-fresh-1' })

    const assistants = (stream.state(SID).messages ?? []).filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(2)
    expect(assistants[1].parts).toEqual([
      expect.objectContaining({ result: 'match', toolCallId: 'call-fresh-1', type: 'tool-call' })
    ])
  })

  it('does not move a completion away from the live stream when it owns the call', () => {
    event('message.start', 500)
    event('tool.start', 501, { args: {}, name: 'terminal', tool_id: 'call-live-1' })
    event('tool.complete', 502, { name: 'terminal', result: 'ok', tool_id: 'call-live-1' })
    event('message.complete', 503, { text: 'Done.' })

    const assistants = (stream.state(SID).messages ?? []).filter(message => message.role === 'assistant')

    // Unchanged single-bubble behaviour for the plain in-stream case.
    expect(assistants).toHaveLength(1)
    expect(assistants[0].parts.map(part => part.type)).toEqual(['tool-call', 'text'])
    expect(assistants[0].parts[0]).toMatchObject({ result: 'ok' })
  })
})
