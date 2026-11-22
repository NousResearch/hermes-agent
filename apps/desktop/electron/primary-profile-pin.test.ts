import assert from 'node:assert/strict'

import { test } from 'vitest'

import { PrimaryProfilePin, resolveLaunchProfile } from './primary-profile-pin'

test('a live primary keeps answering for its booted profile after the preference moves', () => {
  const pin = new PrimaryProfilePin()
  let preference: null | string = 'default'

  // startHermes() boots the primary as the preference at that moment.
  assert.equal(pin.pin(preference), 'default')
  assert.equal(
    pin.resolve(() => preference),
    'default'
  )

  // hermes:profile:remember rewrites active-profile.json without re-homing.
  preference = 'claude'

  // Routing must still see the running primary as "default": otherwise a
  // request for "default" falls through to the pool and a second backend is
  // spawned for the same HERMES_HOME.
  assert.equal(
    pin.resolve(() => preference),
    'default'
  )
})

test('teardown releases the pin so the next start follows the preference', () => {
  const pin = new PrimaryProfilePin()
  pin.pin('default')
  pin.clear()

  assert.equal(pin.booted, null)
  assert.equal(
    pin.resolve(() => 'claude'),
    'claude'
  )
})

// #108417: one authoritative launch-profile decision per startup attempt.
// startHermes used to pin primaryProfileKey() at the top and separately
// re-read the preference deep inside the connection IIFE for --profile and
// the child env — two reads that a mid-startup hermes:profile:remember could
// split into "routing says alpha, argv says beta". resolveLaunchProfile makes
// them ONE read with two encodings of the unset case.
test('one launch decision feeds routing, argv, and env from the same read', () => {
  const named = resolveLaunchProfile(() => 'beta')

  assert.equal(named.routingProfile, 'beta')
  assert.equal(named.argvProfile, 'beta')

  // Unset preference: routing falls back to 'default', the launch argument
  // keeps the legacy shape (no --profile flag at all).
  const unset = resolveLaunchProfile(() => null)

  assert.equal(unset.routingProfile, 'default')
  assert.equal(unset.argvProfile, null)

  const blank = resolveLaunchProfile(() => '   ')

  assert.equal(blank.routingProfile, 'default')
  assert.equal(blank.argvProfile, null)
})
