import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { clearSessionTodos } from '@/store/todos'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

/**
 * Redelivered `message.interim` events must not duplicate bubbles (#93926).
 *
 * The gateway delivers stream events over a reconnecting websocket, so any
 * interim event can arrive twice: back-to-back, or long after its segment was
 * sealed once later commentary has streamed. Redelivery is recognized as a
 * restatement of recent assistant text. Distinct texts must still append
 * (no over-suppression), and `already_streamed` alone must not suppress:
 * upstream fixtures rely on those events appending when no bubble is open
 * (the streamed deltas are gone, so the event is the only copy).
 */

const SID = 'session-1'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

const ev = (type: string, payload: Record<string, unknown> = {}): GatewayEvent =>
  ({ payload, session_id: SID, type }) as GatewayEvent

const start = () => act(() => stream.handleEvent(ev('message.start')))
const delta = (text: string) => act(() => stream.handleEvent(ev('message.delta', { text })))

const interim = (text: string, extra: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent(ev('message.interim', { text, ...extra })))

function assistantTexts(): string[] {
  return stream
    .state(SID)
    .messages.filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
    .filter(Boolean)
}

beforeEach(() => {
  clearSessionTodos(SID)
})

afterEach(() => {
  cleanup()
  clearSessionTodos(SID)
  vi.restoreAllMocks()
})

describe('useMessageStream redelivered message.interim (#93926)', () => {
  it('keeps one bubble when the same interim arrives twice back-to-back', () => {
    mountStream()
    start()
    interim('checking the attach path')
    interim('checking the attach path')

    expect(assistantTexts().filter(t => t === 'checking the attach path')).toHaveLength(1)
  })

  it('keeps one bubble when an earlier interim is replayed after later segments streamed', () => {
    mountStream()
    start()
    interim('first commentary')
    delta('second commentary')
    interim('second commentary')
    // Transport replay of the first event, well after its bubble was sealed:
    interim('first commentary')

    const texts = assistantTexts()
    expect(texts.filter(t => t === 'first commentary')).toHaveLength(1)
    expect(texts).toContain('second commentary')
  })

  it('keeps one bubble for a whitespace-normalized restatement', () => {
    mountStream()
    start()
    interim('checking the attach path')
    interim('checking  the   attach\npath')

    const texts = assistantTexts()
    // Count total bubbles, not matching texts: the raw copy must be dropped,
    // not merely equal to a normalized expectation string.
    expect(texts).toHaveLength(1)
    expect(texts[0]).toBe('checking the attach path')
  })

  it('still appends genuinely new commentary after a suppression', () => {
    mountStream()
    start()
    interim('first commentary')
    interim('first commentary')
    interim('a different observation')

    const texts = assistantTexts()
    expect(texts.filter(t => t === 'first commentary')).toHaveLength(1)
    expect(texts).toContain('a different observation')
  })

  it('does not suppress distinct commentary with a shared prefix', () => {
    mountStream()
    start()
    interim('Checking the composer path for regressions.')
    interim('Checking the composer path for regressions in v0.20.5.')

    const texts = assistantTexts()
    expect(texts.filter(t => t.startsWith('Checking the composer path'))).toHaveLength(2)
  })
})
