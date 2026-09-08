import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'timeline-session'
const hydrateFromStoredSession = vi.fn(async () => undefined)

let stream: MessageStreamHarness

const event = (type: string, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

describe('live transcript timeline events', () => {
  it('starts a typed background boundary only when its follow-up actually starts', () => {
    event('message.start', 10)
    event('message.delta', 11, { text: 'Main answer.' })
    event('message.complete', 12, { text: 'Main answer.' })
    event('status.update', 13, { kind: 'process', text: 'An internal notification is queued.' })
    expect(stream.state(SID).messages.some(message => message.displayKind)).toBe(false)
    event('message.start', 14, { display_kind: 'async_delegation_complete', display_metadata: { task_count: 1 } })
    expect(stream.state(SID).messages.at(-1)?.asyncResult).toBeUndefined()
    event('message.delta', 15, { text: 'Review finished.' })
    event('message.complete', 16, { text: 'Review finished.' })
    expect(stream.state(SID).messages.map(message => [message.role, message.displayKind])).toEqual([
      ['assistant', undefined],
      ['system', 'async_delegation_complete'],
      ['assistant', undefined]
    ])
  })

  it.each([
    ['completed', 'A **useful finding** with [evidence](https://example.com/evidence).'],
    [
      'failed',
      'The subagent did not complete successfully (status=failed).\nWorker failed: run `npm test` to reproduce.'
    ]
  ])('keeps %s worker output through live events before history hydration', async (status, result) => {
    const content = [
      '[ASYNC DELEGATION COMPLETE — deleg_live]',
      'A background subagent you dispatched earlier has finished.',
      'Original goal: Inspect the change',
      `Status: ${status}   API calls: 1   Duration: 1s`,
      '--- RESULT ---',
      result,
      'Full live transcript (complete tool/assistant trace): /tmp/worker-transcript.jsonl'
    ].join('\n')

    const payload = {
      text: content,
      display_kind: 'async_delegation_complete',
      display_metadata: {
        task_count: 1,
        completed_count: status === 'completed' ? 1 : 0,
        failed_count: status === 'failed' ? 1 : 0
      }
    }

    event('message.start', 20, payload)

    const boundary = stream.state(SID).messages[0]
    expect(boundary.asyncResult).toBe(result)
    expect(boundary.role).toBe('system')
    expect(chatMessageText(boundary)).toBe('1 background agent finished')
    expect(toRuntimeMessage(boundary).metadata.custom?.asyncResult).toBe(result)
    expect(payload.text).toBe(content)

    // Even a no-delta failure of the follow-up must not erase the worker result.
    await event('message.complete', 21, {
      status: 'error',
      error: 'Follow-up provider unavailable',
      text: 'Error: Follow-up provider unavailable'
    })
    expect(stream.state(SID).messages[0]).toBe(boundary)
    expect(stream.state(SID).messages.at(-1)?.error).toBeTruthy()
    expect(stream.state(SID).messages.map(chatMessageText).join('\n')).not.toContain('[ASYNC DELEGATION')
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  beforeEach(async () => {
    hydrateFromStoredSession.mockClear()
    stream = renderMessageStream(SID, { hydrateFromStoredSession })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('records commentary, tool, resumed text, and turn-stop boundaries', () => {
    event('message.start', 100)
    event('message.delta', 101.125, { text: 'Let me inspect it.' })
    event('message.interim', 101.75, { already_streamed: true, text: 'Let me inspect it.' })
    event('tool.start', 102.25, { args: { path: 'README.md' }, name: 'read_file', tool_id: 'call-1' })
    event('tool.complete', 104.5, { name: 'read_file', result: { content: 'ok' }, tool_id: 'call-1' })
    event('message.delta', 105.625, { text: 'The file looks good.' })
    event('message.complete', 106.875, { text: 'The file looks good.' })

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
    event('message.start', 200)
    event('reasoning.delta', 201.125, { text: 'Thinking first.' })
    event('message.delta', 202.25, { text: 'Then speaking.' })
    event('tool.start', 203.5, { args: {}, name: 'terminal', tool_id: 'call-2' })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.map(part => part.type)).toEqual(['reasoning', 'text', 'tool-call'])
    expect(assistant?.parts.map(part => part.timestamp)).toEqual([201.125, 202.25, 203.5])
  })

  it('uses the gateway event time for an error boundary', () => {
    event('message.start', 300)
    event('error', 301.875, { error: 'provider failed' })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeTruthy()
    expect([assistant?.timestamp, assistant?.completedAt]).toEqual([301.875, 301.875])
  })

  it('uses the gateway event time for a review summary system row', () => {
    event('review.summary', 401.625, { text: 'Review saved.' })

    const system = stream.state(SID).messages.find(message => message.role === 'system')

    expect(system?.timestamp).toBe(401.625)
    expect(system?.parts[0].timestamp).toBe(401.625)
  })

  it('uses session.info time when it is the only stop boundary', () => {
    event('message.start', 500)
    event('tool.start', 501, { args: {}, name: 'terminal', tool_id: 'call-3' })
    event('session.info', 502.75, { running: false })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.completedAt).toBe(502.75)
    expect(assistant?.parts[0].completedAt).toBe(502.75)
  })
})
