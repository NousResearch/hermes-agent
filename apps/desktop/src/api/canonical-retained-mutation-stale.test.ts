import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// dokterdok N34: a retained (reply-lost) rename must not be replayed for a LATER equal rename.
test('an acknowledged later edit retires an older retained mutation; its stale receipt never rewinds the revision', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7 })

  // Rename to A: the reply is lost (no reason), so the request stays retained for an exact retry.
  const lostA = protocol.prepare('session.title', { session_id: 's', title: 'A' })
  protocol.failure(lostA, new Error('socket closed'))

  // It did apply (revision 8, seen through session.info); then rename to B is acknowledged at 9.
  protocol.event({ type: 'session.info', session_id: 's', payload: { revision: 8 } })
  const renameB = protocol.prepare('session.title', { session_id: 's', title: 'B' })
  expect(renameB.expected_revision).toBe(8)
  protocol.result('session.title', renameB, { session_id: 's', revision: 9, title: 'B' })

  // Renaming back to A is a NEW edit at revision 9, not a replay of the old request id.
  const againA = protocol.prepare('session.title', { session_id: 's', title: 'A' })
  expect(againA.request_id).not.toBe(lostA.request_id)
  expect(againA.expected_revision).toBe(9)

  // An exact-retry receipt replaying the original revision cannot move the CAS value backwards.
  protocol.result('session.title', lostA, { session_id: 's', revision: 8, title: 'A' })
  expect(protocol.prepare('session.archive', { session_id: 's', archived: true }).expected_revision).toBe(9)
})
