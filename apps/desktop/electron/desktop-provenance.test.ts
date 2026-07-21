import { strict as assert } from 'node:assert'

import { test } from 'vitest'

import {
  desktopPublicKeyFingerprint,
  desktopTextHash,
  generateDesktopSigningIdentity,
  signDesktopPayload,
  TrustedGestureLedger,
  verifyDesktopPayload
} from './desktop-provenance'

function payload(text = 'log duty') {
  return {
    version: 1 as const,
    event_id: 'event-1',
    issued_at: '2026-07-21T15:00:00.000Z',
    installation_id: 'install-1',
    os_account: 'darwin:501',
    app_identity: 'TEAM:io.hermes.desktop',
    app_instance_id: 'instance-1',
    profile: 'default',
    window_id: '7',
    session_id: 'session-1',
    text_hash: desktopTextHash(text)
  }
}

test('desktop prompt signatures bind every payload field', () => {
  const keys = generateDesktopSigningIdentity()
  const signed = payload()
  const signature = signDesktopPayload(keys.privateKeyPem, signed)

  assert.equal(verifyDesktopPayload(keys.publicKeyPem, signed, signature), true)
  assert.equal(verifyDesktopPayload(keys.publicKeyPem, { ...signed, profile: 'other' }, signature), false)
  assert.match(desktopPublicKeyFingerprint(keys.publicKeyPem), /^[a-f0-9]{64}$/)
})

test('trusted gesture is one-use and recovery cannot change text or profile', () => {
  const ledger = new TrustedGestureLedger()
  const now = 1_000
  const firstHash = desktopTextHash('first')

  assert.equal(ledger.eventIdFor(7, firstHash, 'default', now), null)
  ledger.note(7, firstHash, now)
  const eventId = ledger.eventIdFor(7, firstHash, 'default', now + 1)

  assert.ok(eventId)
  assert.equal(ledger.eventIdFor(7, firstHash, 'default', now + 2), eventId)
  assert.equal(ledger.eventIdFor(7, desktopTextHash('changed'), 'default', now + 2), null)
  assert.equal(ledger.eventIdFor(7, firstHash, 'other', now + 2), null)
})

test('trusted composer gesture cannot authorize substituted renderer text', () => {
  const ledger = new TrustedGestureLedger()
  const now = 1_000
  const intended = desktopTextHash('log my duty')

  ledger.note(7, intended, now)
  assert.equal(ledger.eventIdFor(7, desktopTextHash('delete everything'), 'default', now + 1), null)
  assert.ok(ledger.eventIdFor(7, intended, 'default', now + 2))
})
