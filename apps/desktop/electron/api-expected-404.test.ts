import assert from 'node:assert/strict'

import { test } from 'vitest'

import { HERMES_API_EXPECTED_404, isExpectedNotFoundSentinel, unwrapExpectedNotFound } from './api-expected-404'

test('a handler-produced sentinel is recognized', () => {
  assert.equal(isExpectedNotFoundSentinel({ [HERMES_API_EXPECTED_404]: '404: {"detail":"Session not found"}' }), true)
})

test('real backend payloads are never mistaken for a sentinel', () => {
  const notSentinels: unknown[] = [
    null,
    undefined,
    'string',
    42,
    [],
    [{ [HERMES_API_EXPECTED_404]: 'x' }],
    {},
    { session_id: 's1' },
    // right key, wrong value type
    { [HERMES_API_EXPECTED_404]: 404 },
    // right key, but carries real data alongside — not our shape
    { [HERMES_API_EXPECTED_404]: 'x', session_id: 's1' }
  ]

  for (const value of notSentinels) {
    assert.equal(isExpectedNotFoundSentinel(value), false, `treated ${JSON.stringify(value)} as a sentinel`)
  }
})

test('unwrap rethrows the sentinel as the exact rejection the renderer expects', () => {
  const message = '404: {"detail":"Session not found"}'

  assert.throws(
    () => unwrapExpectedNotFound({ [HERMES_API_EXPECTED_404]: message }),
    (error: Error) => error instanceof Error && error.message === message
  )
})

test('unwrap passes real responses through untouched, by reference', () => {
  const payload = { session_id: 's1', messages: [] }

  assert.equal(unwrapExpectedNotFound(payload), payload)
  assert.equal(unwrapExpectedNotFound(null), null)
  assert.equal(unwrapExpectedNotFound(''), '')
})

test('the rethrown message still matches the renderer 404 probe predicate', () => {
  // `resolveStoredSession` and friends branch on a `404`-shaped message; the
  // seam must not change what they see.
  const message = '404: {"detail":"Session not found"}'

  try {
    unwrapExpectedNotFound({ [HERMES_API_EXPECTED_404]: message })
    assert.fail('expected a rejection')
  } catch (error) {
    assert.match(String((error as Error).message), /(?:^|\s)404\b/)
  }
})
