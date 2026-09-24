import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'pre-start-stream-fence-session'

const emit = (stream: MessageStreamHarness, type: GatewayEventName, text = '') =>
  act(() => stream.handleEvent({ payload: text ? { text } : {}, session_id: SID, type }))

describe('pre-start stream fence', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('does not append a late prior-turn delta while the next submitted turn is still unconfirmed', async () => {
    vi.useFakeTimers()
    const stream = renderMessageStream(SID)

    emit(stream, 'message.start')
    emit(stream, 'message.delta', 'previous reply')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    emit(stream, 'message.complete', 'previous reply')

    // submitPrompt has armed the new turn, but its message.start has not
    // arrived. A cancelled background-review stream can still have one late
    // delta in flight during exactly this gap.
    stream.states.set(SID, { ...stream.state(), awaitingResponse: true, busy: true, turnLive: false })
    emit(stream, 'message.delta', 'stale review fragment')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    expect(stream.text()).toBe('previous reply')

    emit(stream, 'message.start')
    emit(stream, 'message.delta', 'current reply')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    expect(stream.text()).toBe('current reply')
  })
})
