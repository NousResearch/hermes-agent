import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

// A typed /undo or /retry derives its target from a fresh snapshot. After a lost reply the owner has
// already rewound, so the repeat's snapshot names the turn before: the retained rewind, not a new one at
// that target, must be what the repeat sends (as Ink's mutateCanonical and the classic CLI do).
function primed() {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 7, execution_generation: 3 })

  return protocol
}

const derived = (target: number) => ({ session_id: 's', target_message_id: target, derived_target: true })

test('a derived rewind whose reply was lost is re-sent for the repeat, not re-derived', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(5))
  protocol.failure(lost, new Error('Hermes gateway connection lost'))

  // The repeat's snapshot already lacks row 5 and names row 3; the original request goes out again.
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 8, execution_generation: 4 })
  const repeat = protocol.prepare('session.rewind', derived(3))

  expect(repeat).toEqual(lost)
  expect(repeat).toMatchObject({ payload: { target_message_id: 5 }, expected_revision: 7, expected_generation: 3 })
})

test('an answered derived rewind is not replayed for the next /undo', () => {
  const protocol = primed()
  const first = protocol.prepare('session.rewind', derived(5))
  protocol.result('session.rewind', first, { session_id: 's', revision: 8, rewound_count: 2, target_message: { id: 5 } })

  const next = protocol.prepare('session.rewind', derived(3))

  expect(next.request_id).not.toBe(first.request_id)
  expect(next).toMatchObject({ payload: { target_message_id: 3 }, expected_revision: 8 })
})

test('a typed refusal of a derived rewind lets the repeat derive again', () => {
  const protocol = primed()
  const refused = protocol.prepare('session.rewind', derived(5))
  protocol.failure(refused, { data: { reason: 'revision_conflict', revision: 9 } })

  const retry = protocol.prepare('session.rewind', derived(5))

  expect(retry.request_id).not.toBe(refused.request_id)
  expect(retry).toMatchObject({ payload: { target_message_id: 5 }, expected_revision: 9 })
})

test('another control retires a lost derived rewind', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', derived(5))
  protocol.failure(lost, new Error('socket closed'))

  protocol.prepare('session.title', { session_id: 's', title: 'renamed' })
  const later = protocol.prepare('session.rewind', derived(3))

  expect(later.request_id).not.toBe(lost.request_id)
  expect(later).toMatchObject({ payload: { target_message_id: 3 } })
})

test('an explicit rewind (edit, regenerate, restore) stays keyed by its own target', () => {
  const protocol = primed()
  const lost = protocol.prepare('session.rewind', { session_id: 's', target_message_id: 5 })
  protocol.failure(lost, new Error('socket closed'))

  expect(protocol.prepare('session.rewind', { session_id: 's', target_message_id: 5 })).toEqual(lost)

  const other = protocol.prepare('session.rewind', { session_id: 's', target_message_id: 3 })

  expect(other.request_id).not.toBe(lost.request_id)
  expect(other).toMatchObject({ payload: { target_message_id: 3 } })
})
