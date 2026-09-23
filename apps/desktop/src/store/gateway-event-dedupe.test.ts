import { describe, expect, it } from 'vitest'

import { GatewayEventDeduper } from './gateway-event-dedupe'

describe('GatewayEventDeduper', () => {
  it('delivers a sequenced event from two sockets only once', () => {
    const deduper = new GatewayEventDeduper()
    const frame = { replayEpoch: 'backend-a', seq: 7, session_id: 'session-1' }

    expect(deduper.accept(frame)).toBe(true)
    expect(deduper.accept(frame)).toBe(false)
    expect(deduper.accept({ ...frame, seq: 8 })).toBe(true)
  })

  it('accepts a counter reset and seq-less legacy events', () => {
    const deduper = new GatewayEventDeduper()

    expect(deduper.accept({ replayEpoch: 'backend-a', seq: 2048, session_id: 'session-1' })).toBe(true)
    expect(deduper.accept({ replayEpoch: 'backend-a', seq: 1, session_id: 'session-1' })).toBe(true)
    expect(deduper.accept({ replayEpoch: 'backend-a', session_id: 'session-1' })).toBe(true)
    expect(deduper.accept({ replayEpoch: 'backend-a', seq: 1 })).toBe(true)
  })
})
