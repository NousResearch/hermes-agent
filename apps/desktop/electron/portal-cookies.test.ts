import assert from 'node:assert/strict'

import { test } from 'vitest'

import { cookiesHavePortalAccessToken, cookiesHavePortalSession } from './portal-cookies'

// Contract: NAS accepts either credential family during the auth migration, so
// the Desktop must read both, and must tell renewable-only jars (signed in,
// discovery would 401) apart from jars that can pass /api/agents right now.
test('either portal credential family signs in, and refresh-only material is a session without access', () => {
  for (const [access, refresh] of [
    ['privy-token', 'privy-refresh-token'],
    ['nas-session', 'nas-refresh']
  ]) {
    const accessJar = [{ name: access, value: 'jwt' }]
    const refreshJar = [{ name: refresh, value: 'jwt' }]

    assert.equal(cookiesHavePortalSession(accessJar), true)
    assert.equal(cookiesHavePortalAccessToken(accessJar), true)
    assert.equal(cookiesHavePortalSession(refreshJar), true)
    assert.equal(cookiesHavePortalAccessToken(refreshJar), false)
  }
})

// Contract: the jar also holds Hermes GATEWAY session cookies, NAS provider
// routing hints and logout identifiers. None of those authenticate the portal.
test('gateway cookies, routing hints, empty values and non-arrays are never a portal credential', () => {
  const noise = [
    { name: 'hermes_session_at', value: 'x' },
    { name: '__Host-hermes_session_rt', value: 'x' },
    { name: 'next-auth-provider', value: 'workos' },
    { name: 'workos-session-id', value: 'logout-only' },
    { name: 'privy-token', value: '' },
    { name: 'nas-session', value: '' }
  ]

  for (const jar of [noise, [], null, undefined]) {
    assert.equal(cookiesHavePortalSession(jar), false)
    assert.equal(cookiesHavePortalAccessToken(jar), false)
  }
})
