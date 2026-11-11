import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// A retained control (reply lost) may be retried only as the session's very next control. Once
// another verb reached the session, re-issuing the old control must mint a fresh identity: replaying
// the stale receipt left the session on the intervening value.
test('another control on the session retires a lost-reply control instead of replaying it', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7, execution_generation: 3 })
  const firstA = protocol.prepare('session.title', { session_id: 's', title: 'A' }) // reply lost
  protocol.prepare('session.title', { session_id: 's', title: 'B' })
  const againA = protocol.prepare('session.title', { session_id: 's', title: 'A' })
  expect(againA.request_id).not.toBe(firstA.request_id)
})

test('a prompt submit retires the session lost-reply controls', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7, execution_generation: 3 })
  const firstA = protocol.prepare('session.title', { session_id: 's', title: 'A' }) // reply lost
  protocol.prepare('prompt.submit', { session_id: 's', text: 'go', submission_id: 'x' })
  expect(protocol.prepare('session.title', { session_id: 's', title: 'A' }).request_id).not.toBe(firstA.request_id)
})

test('the immediate retry of the same control keeps its identity', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7, execution_generation: 3 })
  const first = protocol.prepare('session.title', { session_id: 's', title: 'A' })
  expect(protocol.prepare('session.title', { session_id: 's', title: 'A' })).toEqual(first)
})
