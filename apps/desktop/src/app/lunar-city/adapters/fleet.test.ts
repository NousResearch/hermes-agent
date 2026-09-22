import { describe, expect, it } from 'vitest'

import { normalizeRoster } from './fleet'

describe('normalizeRoster', () => {
  it('keeps same-named profiles on different connections as distinct exact-owner entities', () => {
    const normalized = normalizeRoster(
      {
        agents: [
          {
            connectionId: 'ssh-1',
            connectionKind: 'ssh',
            connectionLabel: 'moon relay',
            handle: '@worker-moon',
            profile: 'worker'
          },
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'this Mac',
            handle: '@worker-local',
            profile: 'worker'
          }
        ],
        sources: [
          { connectionId: 'local', kind: 'local', label: 'this Mac', reachable: true },
          { connectionId: 'ssh-1', kind: 'ssh', label: 'moon relay', reachable: true }
        ]
      },
      { observedAt: 42 }
    )

    expect(normalized.entities.map(entity => entity.identity.connectionId)).toEqual(['local', 'ssh-1'])
    expect(new Set(normalized.entities.map(entity => entity.key)).size).toBe(2)
    expect(normalized.sources.map(source => source.source)).toEqual(['fleet:local', 'fleet:ssh-1'])
  })

  it('retains an unreachable source and its last-known profile as stale and unavailable', () => {
    const normalized = normalizeRoster(
      {
        agents: [
          {
            connectionId: 'ssh-1',
            connectionKind: 'ssh',
            connectionLabel: 'moon relay',
            handle: '@worker-moon',
            profile: 'worker'
          }
        ],
        sources: [
          {
            connectionId: 'ssh-1',
            error: 'offline',
            kind: 'ssh',
            label: 'moon relay',
            reachable: false
          }
        ]
      },
      { observedAt: 42 }
    )

    expect(normalized.sources[0]).toMatchObject({ authority: 'stale', error: 'offline', source: 'fleet:ssh-1' })
    expect(normalized.entities[0]).toMatchObject({
      animation: 'unavailable',
      authority: 'stale',
      destination: 'unavailable'
    })
  })

  it('treats a reachable source reporting an error as stale rather than authoritative', () => {
    const normalized = normalizeRoster(
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'this Mac',
            handle: '@worker-local',
            profile: 'worker'
          }
        ],
        sources: [
          {
            connectionId: 'local',
            error: 'refresh retained cached roster',
            kind: 'local',
            label: 'this Mac',
            reachable: true
          }
        ]
      },
      { observedAt: 42 }
    )

    expect(normalized.sources[0]).toMatchObject({ authority: 'stale', error: 'refresh retained cached roster' })
    expect(normalized.entities[0]).toMatchObject({ authority: 'stale', destination: 'unavailable' })
  })
})
