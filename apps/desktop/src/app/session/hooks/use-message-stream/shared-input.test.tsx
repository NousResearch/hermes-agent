import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText, textPart } from '@/lib/chat-messages'
import type { RpcEvent } from '@/types/hermes'

import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const input = (id: string, ref?: string, session = 's'): RpcEvent => ({
  type: 'message.input',
  session_id: session,
  turn: { id: 'run' },
  payload: {
    kind: 'redirect',
    input: { role: 'user', text: 'Same words', display_kind: 'steer' },
    inputs: [{ id, ref }]
  }
})

describe('shared correction observation', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('flushes prior output, inserts each occurrence once, and keeps later output below it', async () => {
    vi.useFakeTimers()
    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 's', turn: { id: 'run' } })
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'Before' } })
      h.handleEvent(input('one'))
      h.handleEvent(input('one')) // replay overlap
      h.handleEvent(input('two')) // identical words, different submission
      h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'After' } })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    expect(h.state().messages.map(m => [m.role, chatMessageText(m)])).toEqual([
      ['assistant', 'Before'],
      ['user', 'Same words'],
      ['user', 'Same words'],
      ['assistant', 'After']
    ])
    expect(h.state().busy).toBe(true)
  })

  it('recognizes its own optimistic row by reference and rejects foreign executions or unscoped input', () => {
    const h = renderMessageStream('s')
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 's', turn: { id: 'run' } })
    })
    h.states.set('s', { ...h.state(), messages: [{ id: 'local-ref', role: 'user', parts: [textPart('Same words')] }] })
    act(() => {
      h.handleEvent(input('own', 'local-ref'))
      h.handleEvent(input('own', 'local-ref'))
      h.handleEvent({ ...input('stale'), turn: { id: 'old-run' } })
      h.handleEvent({ ...input('unscoped'), session_id: undefined })
      h.handleEvent({ ...input('hidden'), payload: { kind: 'redirect', input: null, inputs: [{ id: 'hidden' }] } })
    })
    expect(h.state().messages.map(m => m.id)).toEqual(['local-ref'])
    act(() => {
      h.handleEvent(input('second-occurrence', 'local-ref'))
    })
    expect(h.state().messages.map(chatMessageText)).toEqual(['Same words', 'Same words'])
    // A background session's correction belongs to its own state and never the focused chat.
    act(() => {
      h.handleEvent({ type: 'message.start', session_id: 'background', turn: { id: 'run' } })
      h.handleEvent(input('background', undefined, 'background'))
    })
    expect(h.state().messages).toHaveLength(2)
    expect(h.state('background').messages.map(chatMessageText)).toEqual(['Same words'])
  })
})
