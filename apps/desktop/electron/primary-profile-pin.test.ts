import assert from 'node:assert/strict'

import { test } from 'vitest'

import { PrimaryProfilePin } from './primary-profile-pin'

test('with no live primary, the stored preference (or default) wins', () => {
  const pin = new PrimaryProfilePin()

  assert.equal(pin.resolve(() => null), 'default')
  assert.equal(pin.resolve(() => '  '), 'default')
  assert.equal(pin.resolve(() => 'claude'), 'claude')
})

test('a live primary keeps answering for its booted profile after the preference moves', () => {
  const pin = new PrimaryProfilePin()
  let preference: null | string = 'default'

  // startHermes() boots the primary as the preference at that moment.
  assert.equal(pin.pin(preference), 'default')
  assert.equal(pin.resolve(() => preference), 'default')

  // hermes:profile:remember rewrites active-profile.json without re-homing.
  preference = 'claude'

  // Routing must still see the running primary as "default": otherwise a
  // request for "default" falls through to the pool and a second backend is
  // spawned for the same HERMES_HOME.
  assert.equal(pin.resolve(() => preference), 'default')
})

test('teardown releases the pin so the next start follows the preference', () => {
  const pin = new PrimaryProfilePin()
  pin.pin('default')
  pin.clear()

  assert.equal(pin.booted, null)
  assert.equal(pin.resolve(() => 'claude'), 'claude')
})

test('pinning normalises blank input to default', () => {
  const pin = new PrimaryProfilePin()

  assert.equal(pin.pin(''), 'default')
  assert.equal(pin.pin(undefined), 'default')
  assert.equal(pin.pin(' grok '), 'grok')
})
