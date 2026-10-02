import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  setLiveTurnBackend
} from '@/store/session-states'

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
  clearAllSessionStates()
  $activeSessionId.set(null)
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

it.each(['message.delta', 'message.complete'] as const)(
  'real no-payload watchdog notice is replaced by late %s',
  async type => {
    $activeSessionId.set(SID)
    const states = new Map()

    const unsubscribe = $sessionStates.subscribe(snapshot => {
      if (snapshot[SID]) {states.set(SID, snapshot[SID])}
    })

    const release = setLiveTurnBackend({
      request: (async () => ({ sessions: [{ id: SID, status: 'idle' }] })) as never
    })

    try {
      publishSessionState(SID, {
        ...createClientSessionState(),
        storedSessionId: 'child-stored',
        busy: true,
        awaitingResponse: true,
        turnLive: true,
        streamId: 'empty-placeholder',
        messages: [{ id: 'empty-placeholder', role: 'assistant', parts: [], pending: true }]
      })

      const stream = renderMessageStream(SID, {
        states,
        updateSessionState: (id, updater) => {
          const next = updater($sessionStates.get()[id])
          publishSessionState(id, next)

          return next
        }
      })

      noteSessionEvent(SID)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(LIVE_TURN_EVENT_SILENCE_MS)
      })
      const notice = stream.state().messages[0]
      expect(notice.errorSurface?.code).toBe('no_reply')
      expect(notice.id).not.toBe('empty-placeholder')
      act(() => stream.handleEvent({ type, session_id: SID, payload: { text: 'The child continued and answered.' } }))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(300)
      })
      expect(stream.state().messages).toHaveLength(1)
      expect(chatMessageText(stream.state().messages[0])).toBe('The child continued and answered.')
      expect(stream.state().messages[0].errorSurface).toBeUndefined()
      expect(stream.state().messages[0].error).toBeUndefined()
    } finally {
      release()
      unsubscribe()
    }
  }
)
