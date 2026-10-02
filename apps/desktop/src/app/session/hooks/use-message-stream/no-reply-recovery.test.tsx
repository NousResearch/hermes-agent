import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'child-watch'
const failed = () => ({
  ...createClientSessionState(),
  heartbeatSettledStreamId: 'watched-turn',
  messages: [
    {
      id: 'watched-turn',
      role: 'assistant' as const,
      parts: [],
      pending: false,
      error: 'Hermes ended this turn without a reply.',
      errorSurface: { code: 'no_reply', layer: 'runtime' as const, retryable: true }
    }
  ]
})

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
})

it('late child text clears only the retry card of its heartbeat-settled turn', async () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() =>
    stream.handleEvent({
      type: 'message.delta',
      session_id: SID,
      payload: { text: 'Python call hung; retrying with curl.' }
    })
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(stream.state().messages).toHaveLength(1)
  expect(chatMessageText(stream.state().messages[0])).toContain('retrying with curl')
  expect(stream.state().messages[0].error).toBeUndefined()
  expect(stream.state().messages[0].errorSurface).toBeUndefined()
})

it('late final reply clears the same card', () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() => stream.handleEvent({ type: 'message.complete', session_id: SID, payload: { text: 'Finished.' } }))
  expect(stream.state().messages).toHaveLength(1)
  expect(stream.state().messages[0].errorSurface).toBeUndefined()
  expect(chatMessageText(stream.state().messages[0])).toBe('Finished.')
})

it('a newer turn keeps the earlier no-reply failure', async () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() => stream.handleEvent({ type: 'message.start', session_id: SID, payload: {} }))
  act(() => stream.handleEvent({ type: 'message.delta', session_id: SID, payload: { text: 'A new answer.' } }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(stream.state().messages[0].errorSurface?.code).toBe('no_reply')
  expect(chatMessageText(stream.state().messages[1])).toBe('A new answer.')
})

it('a real terminal error replaces the inferred no-reply card', () => {
  const stream = renderMessageStream(SID, { states: new Map([[SID, failed()]]) })
  act(() =>
    stream.handleEvent({
      type: 'message.complete',
      session_id: SID,
      payload: {
        status: 'error',
        text: 'Error: provider timeout',
        error: 'provider timeout',
        error_surface: { code: 'provider_timeout', layer: 'provider', retryable: true }
      }
    })
  )
  expect(stream.state().messages[0].error).toBe('provider timeout')
  expect(stream.state().messages[0].errorSurface?.code).toBe('provider_timeout')
})
