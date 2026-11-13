import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isOwnerUnavailableRejection } from './owner-unavailable'

test('the named owner_unavailable shape is the owner refusal', () => {
  const error = new Error('503: {"detail":"owner_unavailable: starting: Exclusive maintenance refused: Gateway runtime already owns profile /home/daisha/.hermes. Drain and stop the gateway, then retry."}')

  assert.equal(isOwnerUnavailableRejection(error), true)
})

test('the bare exclusive-maintenance shape (older servers, the pre-fix race) is the owner refusal', () => {
  const error = new Error('409: {"detail":"Exclusive maintenance refused: Gateway runtime already owns profile /home/daisha/.hermes. Drain and stop the gateway, then retry."}')

  assert.equal(isOwnerUnavailableRejection(error), true)
})

test('ordinary failures are not the owner refusal', () => {
  const ordinary: unknown[] = [
    new Error('503: {"detail":"storage_unavailable"}'),
    new Error('409: {"detail":"revision_conflict"}'),
    new Error('409: {"detail":"session_busy"}'),
    new Error('404: {"detail":"Session not found"}'),
    new Error('network timeout'),
    new Error('503: {"detail":"session_authority_unavailable"}'),
    undefined,
    null
  ]

  for (const value of ordinary) {
    assert.equal(isOwnerUnavailableRejection(value), false, `treated ${String(value)} as the owner refusal`)
  }
})
