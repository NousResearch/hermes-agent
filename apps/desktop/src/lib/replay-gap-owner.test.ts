import { describe, expect, it } from 'vitest'

import { eventSourceMatchesOwner, gatewayEventSource } from './replay-gap-owner'

describe('eventSourceMatchesOwner', () => {
  it('matches an exact route only from its own connection, and a bare profile only from the untagged primary', () => {
    const remote = gatewayEventSource({ connectionId: 'backend-a', profile: 'work' })
    const local = gatewayEventSource({ profile: 'work' })

    // Exact owners: same connection + profile, never a colliding id from elsewhere.
    expect(eventSourceMatchesOwner(remote, { connectionId: 'backend-a', profile: 'work' })).toBe(true)
    expect(eventSourceMatchesOwner(remote, { connectionId: 'backend-b', profile: 'work' })).toBe(false)
    expect(eventSourceMatchesOwner(remote, { connectionId: 'backend-a', profile: 'default' })).toBe(false)

    // Profile-only owners are the local pool route: an untagged event for the
    // profile matches, a tagged one must not (same-named remote profiles).
    expect(eventSourceMatchesOwner(local, 'work')).toBe(true)
    expect(eventSourceMatchesOwner(remote, 'work')).toBe(false)

    // No owner resolved at all: never act on an unattributable gap.
    expect(eventSourceMatchesOwner(local, undefined)).toBe(false)
    expect(eventSourceMatchesOwner(local, '')).toBe(false)
  })
})
