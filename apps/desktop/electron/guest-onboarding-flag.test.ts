import assert from 'node:assert/strict'

import { test } from 'vitest'

import { desktopBackendSpawnEnv, guestOnboardingEnabled } from './guest-onboarding'
import { buildSpawnCommand } from './remote-lifecycle'

test('guestOnboardingEnabled: exactly "1" in env or --guest-onboarding on argv turns the free tier on', () => {
  assert.equal(guestOnboardingEnabled([], { HERMES_GUEST_ONBOARDING: '1' }), true)
  assert.equal(guestOnboardingEnabled(['electron', '.', '--guest-onboarding'], {}), true)

  assert.equal(guestOnboardingEnabled([], {}), false)
  assert.equal(guestOnboardingEnabled([], { HERMES_GUEST_ONBOARDING: 'true' }), false)
  assert.equal(guestOnboardingEnabled([], { HERMES_GUEST_ONBOARDING: '0' }), false)
  assert.equal(guestOnboardingEnabled(['electron', '.', '--local'], { HERMES_GUEST_ONBOARDING: '' }), false)
})

test('desktopBackendSpawnEnv stamps the launch decision last and never lets an inherited value leak', () => {
  const base = {
    HERMES_HOME: '/tmp/home',
    HERMES_DESKTOP: '1',
    HERMES_GUEST_ONBOARDING: '1',
    PATH: '/usr/bin'
  }

  const on = desktopBackendSpawnEnv({ ...base, HERMES_GUEST_ONBOARDING: '0' }, true)
  assert.equal(on.HERMES_GUEST_ONBOARDING, '1')

  const off = desktopBackendSpawnEnv(base, false)
  assert.equal(off.HERMES_GUEST_ONBOARDING, '0', 'a stray inherited "1" must not turn the free tier on')

  for (const env of [on, off]) {
    assert.equal(env.HERMES_HOME, base.HERMES_HOME)
    assert.equal(env.HERMES_DESKTOP, base.HERMES_DESKTOP)
    assert.equal(env.PATH, base.PATH)
  }
})

test('remote SSH spawn command carries HERMES_GUEST_ONBOARDING=1 only when the launch decided on', () => {
  const on = buildSpawnCommand('/x/hermes', 'work', { logPath: '~/.hermes/log', guestOnboarding: true })
  assert.match(on, /exec env HERMES_DESKTOP=1 HERMES_GUEST_ONBOARDING=1 /)

  const off = buildSpawnCommand('/x/hermes', 'work', { logPath: '~/.hermes/log', guestOnboarding: false })
  assert.match(off, /exec env HERMES_DESKTOP=1 /)
  assert.doesNotMatch(off, /HERMES_GUEST_ONBOARDING/)

  const unset = buildSpawnCommand('/x/hermes', 'work', { logPath: '~/.hermes/log' })
  assert.doesNotMatch(unset, /HERMES_GUEST_ONBOARDING/)
})
