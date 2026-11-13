import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  HERMES_API_OWNER_UNAVAILABLE,
  isOwnerUnavailableHttpError,
  isOwnerUnavailableSentinel,
  unwrapOwnerUnavailable
} from './api-owner-unavailable'

test('a handler-produced sentinel is recognized', () => {
  assert.equal(isOwnerUnavailableSentinel({ [HERMES_API_OWNER_UNAVAILABLE]: '503: {"detail":"owner_unavailable: starting: …"}' }), true)
})

test('real backend payloads are never mistaken for a sentinel', () => {
  const notSentinels: unknown[] = [
    null,
    undefined,
    'string',
    42,
    [],
    [{ [HERMES_API_OWNER_UNAVAILABLE]: 'x' }],
    {},
    { session_id: 's1' },
    // right key, wrong value type
    { [HERMES_API_OWNER_UNAVAILABLE]: 503 },
    // right key, but carries real data alongside — not our shape
    { [HERMES_API_OWNER_UNAVAILABLE]: 'x', session_id: 's1' }
  ]

  for (const value of notSentinels) {
    assert.equal(isOwnerUnavailableSentinel(value), false, `treated ${JSON.stringify(value)} as a sentinel`)
  }
})

test('the matcher recognizes both wire shapes of the owner refusal', () => {
  const named = new Error('503: {"detail":"owner_unavailable: starting: Exclusive maintenance refused: Gateway runtime already owns profile /home/daisha/.hermes. Drain and stop the gateway, then retry."}')
  const bare = new Error('409: {"detail":"Exclusive maintenance refused: Gateway runtime already owns profile /home/daisha/.hermes. Drain and stop the gateway, then retry."}')

  assert.equal(isOwnerUnavailableHttpError(named), true)
  assert.equal(isOwnerUnavailableHttpError(bare), true)
})

test('the matcher does not swallow ordinary failures', () => {
  const ordinary: unknown[] = [
    new Error('503: {"detail":"storage_unavailable"}'),
    new Error('409: {"detail":"session_busy"}'),
    new Error('404: {"detail":"Session not found"}'),
    new Error('network timeout'),
    undefined,
    null
  ]

  for (const value of ordinary) {
    assert.equal(isOwnerUnavailableHttpError(value), false, `treated ${String(value)} as the owner refusal`)
  }
})

test('unwrap rethrows the sentinel as the exact rejection the renderer expects', () => {
  const message = '503: {"detail":"owner_unavailable: starting: Exclusive maintenance refused"}'

  assert.throws(
    () => unwrapOwnerUnavailable({ [HERMES_API_OWNER_UNAVAILABLE]: message }),
    (error: Error) => error instanceof Error && error.message === message
  )
})

test('unwrap passes real responses through untouched, by reference', () => {
  const payload = { session_id: 's1', messages: [] }

  assert.equal(unwrapOwnerUnavailable(payload), payload)
  assert.equal(unwrapOwnerUnavailable(null), null)
  assert.equal(unwrapOwnerUnavailable(''), '')
})

test('the rethrown message still matches the renderer classifier', () => {
  // src/api/owner-unavailable.ts branches on the same literals; the seam must
  // not change what the classifier sees.
  const message = '503: {"detail":"owner_unavailable: starting: Exclusive maintenance refused"}'

  try {
    unwrapOwnerUnavailable({ [HERMES_API_OWNER_UNAVAILABLE]: message })
    assert.fail('expected a rejection')
  } catch (error) {
    assert.equal(String((error as Error).message), message)
  }
})
