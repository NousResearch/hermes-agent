/**
 * Direct agent roster derivation — pure function, no host access, so the
 * eligibility rules (Two Owner Communication Modes release §3.1) are
 * testable without a live Desktop bridge.
 */

import { describe, expect, it } from 'vitest'

import { deriveDirectAgentRoster } from './agent-chat-roster'

describe('deriveDirectAgentRoster', () => {
  it('marks a reachable, enumerated profile as available', () => {
    const rows = deriveDirectAgentRoster({
      agents: [{ connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'architect', targetProfile: 'architect', handle: 'architect' }],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }],
    })

    expect(rows).toEqual([
      expect.objectContaining({
        connectionId: 'local',
        profile: 'architect',
        targetProfile: 'architect',
        availability: 'available',
      }),
    ])
  })

  it('marks a source with an enumeration error as unavailable, never dropped silently', () => {
    const rows = deriveDirectAgentRoster({
      agents: [],
      sources: [{ connectionId: 'mac-mini', kind: 'ssh', label: 'Mac mini', reachable: false, error: 'connection refused' }],
    })

    expect(rows).toEqual([
      expect.objectContaining({
        connectionId: 'mac-mini',
        availability: 'unavailable',
        reason: expect.stringContaining('connection refused'),
      }),
    ])
  })

  it('never fabricates a row for a source that reported nothing', () => {
    const rows = deriveDirectAgentRoster({ agents: [], sources: [] })

    expect(rows).toEqual([])
  })

  it('does not fall back to a guessed profile name when handle is missing', () => {
    const rows = deriveDirectAgentRoster({
      agents: [{ connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'ops', handle: '' }],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }],
    })

    // The row still exists (profile is the real identity), but its display
    // handle is never invented — it falls back to the profile name itself,
    // never to a filesystem scan or hardcoded label.
    expect(rows[0]).toMatchObject({ profile: 'ops', handle: 'ops' })
  })

  it('preserves a distinct backend targetProfile for an aliased/remote row, never collapsing to profile', () => {
    const rows = deriveDirectAgentRoster({
      agents: [{ connectionId: 'mac-mini', connectionKind: 'remote', connectionLabel: 'Mac mini', profile: 'moxie', targetProfile: 'default', handle: 'moxie' }],
      sources: [{ connectionId: 'mac-mini', kind: 'remote', label: 'Mac mini', reachable: true }],
    })

    expect(rows[0]).toMatchObject({ profile: 'moxie', targetProfile: 'default' })
  })

  it('marks a row unavailable when the roster omits targetProfile — never defaults it to profile (Architect corrective, 2026-09-02, seventh pass)', () => {
    const rows = deriveDirectAgentRoster({
      agents: [{ connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'architect', handle: 'architect' }],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }],
    })

    // The row still surfaces (never silently dropped) but is NOT selectable
    // — a missing backend identity can never be handed to
    // host.openCanonicalAgentChat's exact-descriptor admission.
    expect(rows[0]).toMatchObject({ profile: 'architect', targetProfile: '', availability: 'unavailable' })
  })

  it('marks a row unavailable when the roster reports a blank targetProfile', () => {
    const rows = deriveDirectAgentRoster({
      agents: [{ connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'architect', targetProfile: '   ', handle: 'architect' }],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }],
    })

    expect(rows[0]).toMatchObject({ profile: 'architect', targetProfile: '', availability: 'unavailable' })
  })

  it('deduplicates identical (connectionId, profile) pairs from noisy union rosters', () => {
    const rows = deriveDirectAgentRoster({
      agents: [
        { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'architect', handle: 'architect' },
        { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'architect', handle: 'architect' },
      ],
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }],
    })

    expect(rows).toHaveLength(1)
  })
})
