import type { GatewayEventMap, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  errorPayload,
  messageCompletePayload,
  messageDeltaPayload,
  messageInterimPayload,
  reasoningDeltaPayload,
  reviewSummaryPayload,
  sessionInfoPayload,
  toolCompletePayload,
  toolStartPayload
} from '@/test/contract'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'timeline-session'

let stream: MessageStreamHarness

// The wire carries no event clock — a row's time is its receipt time — so
// each frame is delivered with `Date.now()` pinned to the moment it "arrives".
function event<K extends GatewayEventName>(type: K, receivedAt: number, payload: GatewayEventMap[K]) {
  vi.spyOn(Date, 'now').mockReturnValue(receivedAt * 1000)
  act(() => stream.emit(type, payload, SID))
}

describe('live transcript timeline events', () => {
  beforeEach(async () => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('records commentary, tool, resumed text, and turn-stop boundaries', () => {
    event('message.start', 100, {})
    event('message.delta', 101.125, messageDeltaPayload({ text: 'Let me inspect it.' }))
    event('message.interim', 101.75, messageInterimPayload({ already_streamed: true, text: 'Let me inspect it.' }))
    event('tool.start', 102.25, toolStartPayload({ args: { path: 'README.md' }, name: 'read_file', tool_id: 'call-1' }))
    event('tool.complete', 104.5, toolCompletePayload({ name: 'read_file', result: { content: 'ok' }, tool_id: 'call-1' }))
    event('message.delta', 105.625, messageDeltaPayload({ text: 'The file looks good.' }))
    event('message.complete', 106.875, messageCompletePayload({ text: 'The file looks good.' }))

    const assistants = stream.state(SID).messages.filter(message => message.role === 'assistant') ?? []

    expect(assistants).toHaveLength(2)
    expect([assistants[0].timestamp, assistants[0].completedAt]).toEqual([101.125, 101.75])
    expect(assistants[0].parts.map(part => [part.timestamp, part.completedAt])).toEqual([[101.125, 101.75]])

    expect([assistants[1].timestamp, assistants[1].completedAt]).toEqual([102.25, 106.875])
    expect(assistants[1].parts.map(part => part.type)).toEqual(['tool-call', 'text'])
    expect(assistants[1].parts.map(part => [part.timestamp, part.completedAt])).toEqual([
      [102.25, 104.5],
      [105.625, 106.875]
    ])
  })

  it('preserves cross-channel delta order inside one flush window', () => {
    event('message.start', 200, {})
    event('reasoning.delta', 201.125, reasoningDeltaPayload({ text: 'Thinking first.' }))
    event('message.delta', 202.25, messageDeltaPayload({ text: 'Then speaking.' }))
    event('tool.start', 203.5, toolStartPayload({ args: {}, name: 'terminal', tool_id: 'call-2' }))

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.map(part => part.type)).toEqual(['reasoning', 'text', 'tool-call'])
    expect(assistant?.parts.map(part => part.timestamp)).toEqual([201.125, 202.25, 203.5])
  })

  it('uses the gateway event time for an error boundary', () => {
    event('message.start', 300, {})
    event('error', 301.875, errorPayload({ message: 'provider failed' }))

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeTruthy()
    expect([assistant?.timestamp, assistant?.completedAt]).toEqual([301.875, 301.875])
  })

  it('uses the gateway event time for a review summary system row', () => {
    event('review.summary', 401.625, reviewSummaryPayload({ text: 'Review saved.' }))

    const system = stream.state(SID).messages.find(message => message.role === 'system')

    expect(system?.timestamp).toBe(401.625)
    expect(system?.parts[0].timestamp).toBe(401.625)
  })

  it('uses session.info time when it is the only stop boundary', () => {
    event('message.start', 500, {})
    event('tool.start', 501, toolStartPayload({ args: {}, name: 'terminal', tool_id: 'call-3' }))
    event('session.info', 502.75, sessionInfoPayload({ running: false }))

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.completedAt).toBe(502.75)
    expect(assistant?.parts[0].completedAt).toBe(502.75)
  })
})
