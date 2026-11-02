import { describe, expect, it } from 'vitest'

import { createGatewayEventDedupe, DUPLICATE_WINDOW_MS } from './gateway-event-dedupe'

const delta = (seq: number, extra: Record<string, unknown> = {}) =>
  ({
    payload: { text: 'Two tools' },
    replayEpoch: 'epoch-a',
    seq,
    session_id: 'rt-1',
    type: 'message.delta',
    ...extra
  }) as never

describe('gateway event dedupe (#120005)', () => {
  it('admits a frame once even when a second socket to the same backend delivers it again', () => {
    // The backend stamps `seq` before its transport fan-out, so both sockets
    // carry the same (epoch, session, seq); the second copy must not reach
    // the stores or the streaming text doubles.
    const gate = createGatewayEventDedupe()

    expect(gate.admit(delta(359), 1_000)).toBe(true)
    expect(gate.admit(delta(359, { connectionId: 'local' }), 1_005)).toBe(false)
    expect(gate.admit(delta(360), 1_010)).toBe(true)

    // A different backend process has a different epoch: never a duplicate.
    expect(gate.admit(delta(359, { replayEpoch: 'epoch-b' }), 1_020)).toBe(true)
  })

  it('accepts a re-numbered session after the window, and always passes seq-less events', () => {
    const gate = createGatewayEventDedupe()

    for (let seq = 1; seq <= 5; seq += 1) {
      expect(gate.admit(delta(seq), 1_000 + seq)).toBe(true)
    }

    // The backend restarts the counter at 1 when the session leaves its replay
    // ring; a short session's whole restart would otherwise be swallowed.
    expect(gate.admit(delta(1), 1_010 + DUPLICATE_WINDOW_MS)).toBe(true)

    expect(gate.admit(delta(2, { seq: undefined }), 1_000)).toBe(true)
    expect(gate.admit(delta(2, { seq: undefined }), 1_000)).toBe(true)
    expect(gate.admit({ payload: {}, type: 'skin.changed' } as never, 1_000)).toBe(true)
    expect(gate.admit({ payload: {}, type: 'skin.changed' } as never, 1_000)).toBe(true)
  })
})
