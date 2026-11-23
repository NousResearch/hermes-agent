import { describe, expect, it } from 'vitest'

import { createProfileBackendGrants, PROFILE_BACKEND_GRANT_TTL_MS } from './profile-backend-grants'

describe('profile backend grants', () => {
  it('a grant replaces exactly one confirm for the action, plugin and profile it names', () => {
    const grants = createProfileBackendGrants()
    const grant = grants.mint('clear', 'acme', 'work')

    expect(grants.redeem(grant, 'clear', 'acme', 'work')).toBe(true)
    expect(grants.redeem(grant, 'clear', 'acme', 'work')).toBe(false)
  })

  it('refuses another plugin, profile or action and spends the grant anyway', () => {
    const grants = createProfileBackendGrants()
    const forOther = grants.mint('clear', 'acme', 'work')

    expect(grants.redeem(forOther, 'clear', 'acme', 'notes')).toBe(false)
    expect(grants.redeem(forOther, 'clear', 'acme', 'work')).toBe(false)

    expect(grants.redeem(grants.mint('clear', 'acme', 'work'), 'clear', 'other-plugin', 'work')).toBe(false)
    expect(grants.redeem(grants.mint('set', 'acme', 'work'), 'clear', 'acme', 'work')).toBe(false)
    expect(grants.redeem(grants.mint('clear', 'acme', 'work'), 'set', 'acme', 'work')).toBe(false)
    expect(grants.redeem(grants.mint('set', 'acme', 'work'), 'set', 'acme', 'work')).toBe(true)
    expect(grants.size()).toBe(0)
  })

  it('expires', () => {
    let clock = 1_000
    const grants = createProfileBackendGrants(() => clock)
    const grant = grants.mint('set', 'acme', 'work')

    clock += PROFILE_BACKEND_GRANT_TTL_MS + 1

    expect(grants.redeem(grant, 'set', 'acme', 'work')).toBe(false)
  })

  it('refuses anything that is not a minted grant', () => {
    const grants = createProfileBackendGrants()
    grants.mint('clear', 'acme', 'work')

    for (const presented of [undefined, null, '', 42, {}, 'x'.repeat(43)]) {
      expect(grants.redeem(presented, 'clear', 'acme', 'work')).toBe(false)
    }

    expect(grants.size()).toBe(1)
  })
})
