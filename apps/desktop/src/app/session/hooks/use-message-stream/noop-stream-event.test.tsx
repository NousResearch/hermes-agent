import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
let stream: MessageStreamHarness

function emit(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useMessageStream no-op stream events', () => {
  it('keeps the session state when a reasoning replace lands after answer text', () => {
    stream = renderMessageStream(SID)

    emit('message.start')
    emit('message.delta', { text: 'The answer.' })
    // A replace event flushes the throttled text queue first, so the bubble
    // holds the answer after it.
    emit('reasoning.available', { text: 'warmup' })

    const settled = stream.state()

    expect(stream.text()).toBe('The answer.')

    // Answer text already exists, so a later replace keeps the parts as they are.
    emit('reasoning.available', { text: 'late summary' })

    expect(stream.state()).toBe(settled)
  })
})
