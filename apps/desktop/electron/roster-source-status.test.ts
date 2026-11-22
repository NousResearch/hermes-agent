import { expect, it } from 'vitest'

import { rosterSourceStatus } from './roster-source-status'

it('does not report stale cached profiles as a reachable or signed-in gateway', () => {
  expect(rosterSourceStatus({ profiles: ['default'], error: 'OAuth expired', needsSignIn: true })).toEqual({
    reachable: false,
    error: 'OAuth expired',
    needsSignIn: true
  })
  expect(rosterSourceStatus({ profiles: ['default'], error: 'timed out' }).reachable).toBe(false)
  expect(rosterSourceStatus({ profiles: ['default'] }).reachable).toBe(true)
})
