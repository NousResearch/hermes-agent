import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// C20: a delayed generation-1 session.info replay (session.events.since) with `pending: []` must not
// delete the unknown admission a live generation-2 frame published; a current empty frame still does.
test('a stale session.info replay leaves the live Discard action usable', () => {
  const protocol = new CanonicalDesktopProtocol()
  const discard = { session_id: 's', admission_id: 'lost' }
  protocol.event({
    type: 'session.info',
    session_id: 's',
    payload: {
      execution_generation: 2,
      pending: [{ admission_id: 'lost', status: 'unknown', execution_generation: 2 }]
    }
  })
  protocol.event({ type: 'session.info', session_id: 's', payload: { execution_generation: 1, pending: [] } })
  expect(protocol.prepare('prompt.resolve_unknown', discard)).toEqual({ ...discard, execution_generation: 2 })
  protocol.event({ type: 'session.info', session_id: 's', payload: { execution_generation: 2, pending: [] } })
  expect(() => protocol.prepare('prompt.resolve_unknown', discard)).toThrow('unknown')
})
