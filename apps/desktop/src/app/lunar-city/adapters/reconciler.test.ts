import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'
import { _resetFleetRosterForTests } from '@/store/fleet-roster'
import type { SessionInfo } from '@/types/hermes'

import { entityKey } from '../identity'
import { $lunarCitySnapshot, createLunarCitySnapshot, type LunarDelta } from '../store'

import { createLunarCityReconciler, shouldReconcile, startLunarCityReconciler } from './reconciler'

const profileIdentity = { connectionId: 'local', kind: 'profile' as const, profile: 'worker' }

const profile = {
  animation: 'rest',
  authority: 'authoritative' as const,
  destination: 'garden' as const,
  identity: profileIdentity,
  key: entityKey(profileIdentity),
  observedAt: 0
}

const currentRead = () => ({
  authoritative: true,
  entities: [profile],
  sources: [{ authority: 'authoritative' as const, observedAt: 0, source: 'fleet:local' }]
})

async function flush(): Promise<void> {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve()
  }
}

afterEach(() => {
  vi.useRealTimers()
  $lunarCitySnapshot.set(createLunarCitySnapshot())
})

describe('Lunar City reconciler', () => {
  it('recognizes only skipped backend revisions as a reconciliation gap', () => {
    expect(shouldReconcile(4, 6)).toBe(true)
    expect(shouldReconcile(4, 5)).toBe(false)
    expect(shouldReconcile(4, 4)).toBe(false)
  })

  it('tracks backend sequence cursors per exact source rather than globally', () => {
    const reconciler = createLunarCityReconciler({ read: async () => currentRead() })
    const local = { connectionId: 'local', profile: 'worker', sessionId: 'session-a' }
    const remote = { connectionId: 'ssh-1', profile: 'worker', sessionId: 'session-a' }

    expect(reconciler.acceptEvent({ sequence: 5, source: local })).toBe('accepted')
    expect(reconciler.acceptEvent({ sequence: 5, source: remote })).toBe('accepted')
    expect(reconciler.acceptEvent({ sequence: 5, source: local })).toBe('ignored')
  })

  it('coalesces gaps, reconnects, and focus invalidations into one queued reread', async () => {
    let readCount = 0
    let resolveFirst!: (value: ReturnType<typeof currentRead>) => void

    const first = new Promise<ReturnType<typeof currentRead>>(resolve => {
      resolveFirst = resolve
    })

    const reconciler = createLunarCityReconciler({
      read: async () => {
        readCount += 1

        return readCount === 1 ? first : currentRead()
      }
    })

    const stop = reconciler.start()
    await flush()
    reconciler.acceptEvent({
      sequence: 3,
      source: { connectionId: 'local', profile: 'worker', sessionId: 'session-a' }
    })
    reconciler.acceptEvent({
      sequence: 5,
      source: { connectionId: 'local', profile: 'worker', sessionId: 'session-a' }
    })
    reconciler.invalidate('gateway.ready')
    reconciler.invalidate('focus')
    resolveFirst(currentRead())
    await flush()
    await flush()

    expect(readCount).toBe(2)
    stop()
  })

  it('marks retained identity stale at the single freshness deadline', async () => {
    vi.useFakeTimers()
    let now = 0
    const reconciler = createLunarCityReconciler({ freshnessMs: 50, now: () => now, read: async () => currentRead() })

    const stop = reconciler.start()
    await flush()
    now = 50
    await vi.advanceTimersByTimeAsync(50)

    expect($lunarCitySnapshot.get().entities.get(profile.key)).toMatchObject({
      animation: 'unavailable',
      authority: 'stale',
      destination: 'unknown'
    })
    expect($lunarCitySnapshot.get().sources[0]).toMatchObject({ authority: 'stale', observedAt: 0 })
    stop()
  })

  it('stales only expired sources and reschedules the next source deadline', async () => {
    vi.useFakeTimers()
    let now = 0
    const sessionIdentity = { connectionId: 'ssh-1', kind: 'session' as const, profile: 'worker', sessionId: 's-1' }

    const sessionEntity = {
      animation: 'work',
      authority: 'authoritative' as const,
      destination: 'project' as const,
      identity: sessionIdentity,
      key: entityKey(sessionIdentity),
      observedAt: 50
    }

    const reconciler = createLunarCityReconciler({
      freshnessMs: 50,
      now: () => now,
      read: async () => ({
        authoritative: true,
        entities: [profile, sessionEntity],
        sources: [
          { authority: 'authoritative' as const, observedAt: 0, source: 'fleet:local' },
          { authority: 'authoritative' as const, observedAt: 50, source: 'session:ssh-1:worker' }
        ]
      })
    })

    const stop = reconciler.start()
    await flush()
    now = 50
    await vi.advanceTimersByTimeAsync(50)

    expect($lunarCitySnapshot.get().entities.get(profile.key)).toMatchObject({ authority: 'stale' })
    expect($lunarCitySnapshot.get().entities.get(sessionEntity.key)).toMatchObject({ authority: 'authoritative' })
    expect($lunarCitySnapshot.get().sources).toEqual([
      expect.objectContaining({ authority: 'stale', source: 'fleet:local' }),
      expect.objectContaining({ authority: 'authoritative', source: 'session:ssh-1:worker' })
    ])

    now = 100
    await vi.advanceTimersByTimeAsync(50)
    expect($lunarCitySnapshot.get().entities.get(sessionEntity.key)).toMatchObject({ authority: 'stale' })
    stop()
  })

  it('merges source health from a partial read instead of dropping unaffected sources', async () => {
    const sessionIdentity = { connectionId: 'ssh-1', kind: 'session' as const, profile: 'worker', sessionId: 's-1' }

    const sessionEntity = {
      animation: 'work',
      authority: 'authoritative' as const,
      destination: 'project' as const,
      identity: sessionIdentity,
      key: entityKey(sessionIdentity),
      observedAt: 42
    }

    let reads = 0

    const reconciler = createLunarCityReconciler({
      read: async () => {
        reads += 1

        return reads === 1
          ? {
              authoritative: true,
              entities: [profile, sessionEntity],
              sources: [
                { authority: 'authoritative' as const, observedAt: 42, source: 'fleet:local' },
                { authority: 'authoritative' as const, observedAt: 42, source: 'session:ssh-1' }
              ]
            }
          : {
              authoritative: false,
              entities: [
                { ...profile, animation: 'unavailable', authority: 'stale' as const, destination: 'unknown' as const }
              ],
              sources: [{ authority: 'stale' as const, error: 'fleet failed', observedAt: 42, source: 'fleet:local' }]
            }
      }
    })

    const stop = reconciler.start()
    await flush()
    reconciler.invalidate('focus')
    await flush()

    expect($lunarCitySnapshot.get().entities.get(sessionEntity.key)).toMatchObject({ authority: 'authoritative' })
    expect($lunarCitySnapshot.get().sources).toEqual([
      expect.objectContaining({ authority: 'stale', source: 'fleet:local' }),
      expect.objectContaining({ authority: 'authoritative', source: 'session:ssh-1' })
    ])
    stop()
  })

  it('stales unlisted prior entities in an overflowed source namespace', async () => {
    const identity = { board: 'main', connectionId: 'removed', kind: 'kanban' as const, profile: 'worker', taskId: 't' }

    const entity = {
      animation: 'work' as const,
      authority: 'authoritative' as const,
      destination: 'project' as const,
      identity,
      key: entityKey(identity),
      observedAt: 1
    }

    let overflow = false

    const reconciler = createLunarCityReconciler({
      now: () => 2,
      read: async () =>
        overflow
          ? {
              authoritative: false,
              entities: [],
              sources: [
                {
                  authority: 'partial' as const,
                  error: 'Removal tombstone limit exceeded',
                  observedAt: 2,
                  source: 'kanban-registry:removal-overflow'
                }
              ],
              staleUnlistedSourcePrefixes: ['kanban:']
            }
          : {
              authoritative: true,
              entities: [entity],
              sources: [{ authority: 'authoritative' as const, observedAt: 1, source: 'kanban:removed:worker' }]
            }
    })

    const stop = reconciler.start()
    await flush()
    overflow = true
    reconciler.invalidate('overflow')
    await flush()

    expect($lunarCitySnapshot.get().entities.get(entity.key)).toMatchObject({
      animation: 'unavailable',
      authority: 'stale',
      destination: 'unknown'
    })
    stop()
  })

  it('freezes the complete delta payload before handing it to a publisher', async () => {
    let published: LunarDelta | undefined

    const reconciler = createLunarCityReconciler({
      publish: delta => {
        published = delta
      },
      read: async () => currentRead()
    })

    const stop = reconciler.start()
    await flush()

    expect(Object.isFrozen(published)).toBe(true)
    expect(Object.isFrozen(published!.upserts)).toBe(true)
    expect(Object.isFrozen(published!.upserts[0]!)).toBe(true)
    expect(Object.isFrozen(published!.upserts[0]!.identity)).toBe(true)
    expect(Object.isFrozen(published!.sources)).toBe(true)
    expect(Object.isFrozen(published!.sources[0]!)).toBe(true)
    expect(() =>
      (published!.upserts as LunarDelta['upserts'] & { push: (value: typeof profile) => void }).push(profile)
    ).toThrow()
    stop()
  })

  it('accepts restarted lower sequence values after an exact-source reset without resetting a colliding source', () => {
    const reconciler = createLunarCityReconciler({ read: async () => currentRead() })
    const restarted = { connectionId: 'a/b', profile: 'c', sessionId: 'd' }
    const untouched = { connectionId: 'a', profile: 'b/c', sessionId: 'd' }

    expect(reconciler.acceptEvent({ sequence: 5, source: restarted })).toBe('accepted')
    expect(reconciler.acceptEvent({ sequence: 5, source: untouched })).toBe('accepted')
    reconciler.resetSequences({ connectionId: 'a/b', profile: 'c' })
    expect(reconciler.acceptEvent({ sequence: 1, source: restarted })).toBe('accepted')
    expect(reconciler.acceptEvent({ sequence: 1, source: untouched })).toBe('ignored')
  })

  it('clears the affected cursor on gateway.ready so a restarted lower event is reread', async () => {
    const fleet = atom({
      agents: [],
      sources: [{ connectionId: 'local', kind: 'local' as const, label: 'this Mac', reachable: true }]
    })

    let emit!: (event: unknown) => void
    let reads = 0

    const stop = startLunarCityReconciler({
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        refreshFleet: async () => {
          reads += 1

          return { status: 'retained' }
        }
      }
    })

    await flush()
    emit({ connectionId: 'local', profile: 'worker', seq: 5, session_id: 'session-a', type: 'session.changed' })
    await flush()
    emit({ connectionId: 'local', profile: 'worker', type: 'gateway.ready' })
    await flush()
    emit({ connectionId: 'local', profile: 'worker', seq: 1, session_id: 'session-a', type: 'session.changed' })
    await flush()

    expect(reads).toBe(4)
    stop()
  })

  it('prevents a late read from publishing after disposal', async () => {
    let resolveRead!: (value: ReturnType<typeof currentRead>) => void

    const delayed = new Promise<ReturnType<typeof currentRead>>(resolve => {
      resolveRead = resolve
    })

    const reconciler = createLunarCityReconciler({ read: async () => delayed })

    const stop = reconciler.start()
    await flush()
    stop()
    resolveRead(currentRead())
    await flush()

    expect($lunarCitySnapshot.get().revision).toBe(0)
  })

  it('forces one fleet read at route mount without stamping retained data fresh after a failed refresh', async () => {
    let now = 42

    const fleet = atom({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local' as const,
          connectionLabel: 'this Mac',
          handle: '@worker-local',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'local', kind: 'local' as const, label: 'this Mac', reachable: true }]
    })

    const sessions = atom([])
    const subagents = atom({})
    let refreshes = 0
    let focus!: () => void

    const stop = startLunarCityReconciler({
      now: () => now,
      sources: {
        $fleetRoster: fleet,
        $sessions: sessions,
        $subagentsBySession: subagents,
        legacySingleBackend: () => false,
        onFocus: listener => {
          focus = listener

          return () => undefined
        },
        refreshFleet: async () => {
          refreshes += 1

          if (refreshes === 1) {
            fleet.set({ ...fleet.get() })

            return
          }

          return { status: 'failed' } as never
        }
      }
    })

    await flush()

    expect($lunarCitySnapshot.get().sources).toEqual([
      { authority: 'authoritative', observedAt: 42, source: 'fleet:local' }
    ])
    now = 100
    focus()
    await flush()

    expect(refreshes).toBe(2)
    expect($lunarCitySnapshot.get().sources[0]).toMatchObject({
      authority: 'stale',
      error: 'Fleet refresh failed',
      observedAt: 42
    })
    expect($lunarCitySnapshot.get().entities.get(profile.key)).toMatchObject({ authority: 'stale' })
    stop()
  })

  it('hydrates presentation through one exact-source profiles.list route without an active-profile fallback', async () => {
    const fleet = atom({
      agents: [
        {
          connectionId: 'remote-a',
          connectionKind: 'ssh' as const,
          connectionLabel: 'Moon Relay',
          handle: '@scientific-validator',
          profile: 'scientific-validator'
        }
      ],
      sources: [{ connectionId: 'remote-a', kind: 'ssh' as const, label: 'Moon Relay', reachable: true }]
    })

    const readProfileRoster = vi.fn(async (connectionId: string, profileName: string) => ({
      profiles: [
        {
          name: profileName,
          ui_meta: { 'hermes-bots': { groups: ['Research Lab'], title: 'Scientific Validator' } }
        }
      ],
      source: connectionId
    }))

    const sessions = atom([])
    let emit!: (event: unknown) => void

    const stop = startLunarCityReconciler({
      now: () => 42,
      sources: {
        $fleetRoster: fleet,
        $sessions: sessions,
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readProfileRoster,
        refreshFleet: async () => ({ observedAt: 42, status: 'refreshed' })
      }
    })

    await flush()
    await flush()

    expect(readProfileRoster).toHaveBeenCalledOnce()
    expect(readProfileRoster).toHaveBeenCalledWith('remote-a', 'scientific-validator')
    expect([...$lunarCitySnapshot.get().entities.values()][0]).toMatchObject({
      destination: 'lab',
      identity: { connectionId: 'remote-a', profile: 'scientific-validator' },
      presentation: {
        configuredTitle: 'Scientific Validator',
        profileHandle: '@scientific-validator',
        sourceLabel: 'Moon Relay'
      }
    })

    sessions.set([{ id: 'unrelated-session' }] as never)
    await flush()
    await flush()
    emit({
      connectionId: 'remote-a',
      profile: 'scientific-validator',
      seq: 1,
      session_id: 'unrelated-session',
      type: 'session.changed'
    })
    await flush()
    await flush()
    expect(readProfileRoster).toHaveBeenCalledOnce()

    emit({ connectionId: 'remote-a', profile: 'scientific-validator', type: 'gateway.ready' })
    await flush()
    await flush()
    expect(readProfileRoster).toHaveBeenCalledTimes(2)

    fleet.set({ ...fleet.get() })
    await flush()
    await flush()
    expect(readProfileRoster).toHaveBeenCalledTimes(3)
    stop()
  })

  it('retains last-known bot presentation when a metadata refresh fails without changing fleet authority', async () => {
    const fleet = atom({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local' as const,
          connectionLabel: 'This device',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'local', kind: 'local' as const, label: 'This device', reachable: true }]
    })

    let emit!: (event: unknown) => void

    const readProfileRoster = vi
      .fn()
      .mockResolvedValueOnce({
        profiles: [
          {
            name: 'worker',
            ui_meta: { 'hermes-bots': { groups: ['Engineering Guild'], title: 'Builder' } }
          }
        ]
      })
      .mockRejectedValueOnce(new Error('metadata offline'))

    const stop = startLunarCityReconciler({
      now: () => 42,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readProfileRoster,
        refreshFleet: async () => ({ observedAt: 42, status: 'refreshed' })
      }
    })

    await flush()
    await flush()
    emit({ connectionId: 'local', profile: 'worker', type: 'gateway.ready' })
    await flush()
    await flush()

    expect(readProfileRoster).toHaveBeenCalledTimes(2)
    expect([...$lunarCitySnapshot.get().entities.values()][0]).toMatchObject({
      authority: 'authoritative',
      destination: 'project',
      presentation: {
        configuredTitle: 'Builder',
        groups: [{ name: 'Engineering Guild' }],
        metadata: { observedAt: 42, source: 'profiles:local', state: 'stale' }
      }
    })
    stop()
  })

  it('retains a live exact profile slot when an earlier colliding profile joins the roster', async () => {
    const makeRoster = (profiles: readonly string[]): DesktopAgentRoster => ({
      agents: profiles.map(profileName => ({
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        handle: `@${profileName}`,
        profile: profileName
      })),
      sources: [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }]
    })

    const fleet = atom(makeRoster(['worker-004592']))

    const stop = startLunarCityReconciler({
      now: () => 42,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        readProfileRoster: async () => ({ profiles: [] }),
        refreshFleet: async () => ({ observedAt: 42, status: 'refreshed' })
      }
    })

    await flush()
    await flush()
    const before = [...$lunarCitySnapshot.get().entities.values()][0]

    fleet.set(makeRoster(['worker-000729', 'worker-004592']))
    await flush()
    await flush()
    await flush()

    const after = [...$lunarCitySnapshot.get().entities.values()].find(
      entity => entity.identity.kind === 'profile' && entity.identity.profile === 'worker-004592'
    )

    expect(before?.presentation?.placement.slot).toBe(39024)
    expect(after?.presentation?.placement.slot).toBe(39024)
    expect(after?.position).toEqual(before?.position)
    stop()
  })

  it('does not let an in-flight metadata read consume a newer invalidation generation', async () => {
    const fleet = atom({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local' as const,
          connectionLabel: 'This device',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'local', kind: 'local' as const, label: 'This device', reachable: true }]
    })

    let emit!: (event: unknown) => void
    let resolveFirst!: (value: unknown) => void

    const first = new Promise(resolve => {
      resolveFirst = resolve
    })

    const readProfileRoster = vi
      .fn()
      .mockReturnValueOnce(first)
      .mockResolvedValueOnce({
        profiles: [{ name: 'worker', ui_meta: { 'hermes-bots': { title: 'New title' } } }]
      })

    const stop = startLunarCityReconciler({
      now: () => 42,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readProfileRoster,
        refreshFleet: async () => ({ observedAt: 42, status: 'refreshed' })
      }
    })

    await flush()
    expect(readProfileRoster).toHaveBeenCalledOnce()

    emit({ connectionId: 'local', profile: 'worker', type: 'gateway.ready' })
    resolveFirst({ profiles: [{ name: 'worker', ui_meta: { 'hermes-bots': { title: 'Old title' } } }] })
    await flush()
    await flush()
    await flush()

    expect(readProfileRoster).toHaveBeenCalledTimes(2)
    expect([...$lunarCitySnapshot.get().entities.values()][0]?.presentation?.configuredTitle).toBe('New title')
    stop()
  })

  it('retains source-owned metadata across a representative-profile change when refresh fails', async () => {
    let now = 42

    const initial = {
      agents: ['alpha', 'beta'].map(profileName => ({
        connectionId: 'local',
        connectionKind: 'local' as const,
        connectionLabel: 'This device',
        handle: `@${profileName}`,
        profile: profileName
      })),
      sources: [{ connectionId: 'local', kind: 'local' as const, label: 'This device', reachable: true }]
    }

    const fleet = atom(initial)

    const readProfileRoster = vi
      .fn()
      .mockResolvedValueOnce({
        profiles: [
          { name: 'alpha', ui_meta: { 'hermes-bots': { title: 'Alpha' } } },
          { name: 'beta', ui_meta: { 'hermes-bots': { groups: ['Engineering Guild'], title: 'Beta Builder' } } }
        ]
      })
      .mockRejectedValueOnce(new Error('metadata refresh failed'))

    const stop = startLunarCityReconciler({
      now: () => now,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        readProfileRoster,
        refreshFleet: async () => ({ observedAt: now, status: 'refreshed' })
      }
    })

    await flush()
    await flush()
    now = 100
    fleet.set({ ...initial, agents: initial.agents.filter(agent => agent.profile === 'beta') })
    await flush()
    await flush()
    await flush()

    expect(readProfileRoster.mock.calls).toEqual([
      ['local', 'alpha'],
      ['local', 'beta']
    ])
    expect([...$lunarCitySnapshot.get().entities.values()][0]).toMatchObject({
      destination: 'project',
      identity: { connectionId: 'local', profile: 'beta' },
      presentation: {
        configuredTitle: 'Beta Builder',
        metadata: { observedAt: 42, source: 'profiles:local', state: 'stale' }
      }
    })
    stop()
  })

  it('marks retained roster data stale when the real fleet refresh helper catches an enumeration rejection', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(42)
    _resetFleetRosterForTests()
    const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
    const priorDesktop = desktopWindow.hermesDesktop

    const roster: DesktopAgentRoster = {
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'this Mac',
          handle: '@worker-local',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'local', kind: 'local', label: 'this Mac', reachable: true }]
    }

    const getAgentRoster = vi
      .fn<() => Promise<DesktopAgentRoster>>()
      .mockResolvedValueOnce(roster)
      .mockRejectedValueOnce(new Error('gateway asleep'))

    desktopWindow.hermesDesktop = { getAgentRoster } as unknown as Window['hermesDesktop']
    let stop: (() => void) | undefined

    try {
      stop = startLunarCityReconciler()
      await flush()
      await flush()
      await vi.advanceTimersByTimeAsync(0)
      expect(getAgentRoster).toHaveBeenCalledTimes(1)
      expect($lunarCitySnapshot.get().sources).toContainEqual({
        authority: 'authoritative',
        observedAt: 42,
        source: 'fleet:local'
      })

      // Expire the fleet helper's cache without advancing the reconciler
      // freshness timer. The focus read now crosses the production helper's
      // rejection-catching path rather than an injected test seam.
      vi.setSystemTime(60_043)
      window.dispatchEvent(new Event('focus'))
      await flush()
      await flush()
      await vi.advanceTimersByTimeAsync(0)

      expect(getAgentRoster).toHaveBeenCalledTimes(2)
      expect($lunarCitySnapshot.get().sources).toContainEqual({
        authority: 'stale',
        error: 'Fleet roster refresh failed',
        observedAt: 42,
        source: 'fleet:local'
      })
      expect($lunarCitySnapshot.get().entities.get(entityKey(profileIdentity))).toMatchObject({ authority: 'stale' })
    } finally {
      stop?.()
      _resetFleetRosterForTests()

      if (priorDesktop) {
        desktopWindow.hermesDesktop = priorDesktop
      } else {
        delete desktopWindow.hermesDesktop
      }
    }
  })

  it('treats a resolved mixed fleet roster as partial and retains rows from its unreachable source', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(42)
    _resetFleetRosterForTests()
    const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
    const priorDesktop = desktopWindow.hermesDesktop

    const fullRoster: DesktopAgentRoster = {
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'this Mac',
          handle: '@worker-local',
          profile: 'worker'
        },
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'this Mac',
          handle: '@scout-local',
          profile: 'scout'
        },
        {
          connectionId: 'ssh-1',
          connectionKind: 'ssh',
          connectionLabel: 'moon relay',
          handle: '@worker-moon',
          profile: 'worker'
        },
        {
          connectionId: 'ssh-2',
          connectionKind: 'ssh',
          connectionLabel: 'sun relay',
          handle: '@worker-sun',
          profile: 'worker'
        }
      ],
      sources: [
        { connectionId: 'local', kind: 'local', label: 'this Mac', reachable: true },
        { connectionId: 'ssh-1', kind: 'ssh', label: 'moon relay', reachable: true },
        { connectionId: 'ssh-2', kind: 'ssh', label: 'sun relay', reachable: true }
      ]
    }

    const partialRoster: DesktopAgentRoster = {
      agents: [fullRoster.agents[0]!],
      sources: [
        { connectionId: 'local', kind: 'local', label: 'this Mac', reachable: true },
        { connectionId: 'ssh-1', kind: 'ssh', label: 'moon relay', reachable: false },
        { connectionId: 'ssh-2', error: 'degraded', kind: 'ssh', label: 'sun relay', reachable: true }
      ]
    }

    const getAgentRoster = vi
      .fn<() => Promise<DesktopAgentRoster>>()
      .mockResolvedValueOnce(fullRoster)
      .mockResolvedValueOnce(partialRoster)

    desktopWindow.hermesDesktop = { getAgentRoster } as unknown as Window['hermesDesktop']
    let stop: (() => void) | undefined
    const localKey = entityKey({ connectionId: 'local', kind: 'profile', profile: 'worker' })
    const removedLocalKey = entityKey({ connectionId: 'local', kind: 'profile', profile: 'scout' })
    const unreachableKey = entityKey({ connectionId: 'ssh-1', kind: 'profile', profile: 'worker' })
    const erroredKey = entityKey({ connectionId: 'ssh-2', kind: 'profile', profile: 'worker' })

    try {
      stop = startLunarCityReconciler()
      await flush()
      await flush()
      await vi.advanceTimersByTimeAsync(0)

      vi.setSystemTime(60_043)
      window.dispatchEvent(new Event('focus'))
      await flush()
      await flush()
      await vi.advanceTimersByTimeAsync(0)

      expect(getAgentRoster).toHaveBeenCalledTimes(2)
      expect($lunarCitySnapshot.get().entities.get(localKey)).toMatchObject({
        authority: 'authoritative',
        observedAt: 60_043
      })
      expect($lunarCitySnapshot.get().entities.has(removedLocalKey)).toBe(false)
      expect($lunarCitySnapshot.get().entities.get(unreachableKey)).toMatchObject({
        animation: 'unavailable',
        authority: 'stale',
        observedAt: 42
      })
      expect($lunarCitySnapshot.get().entities.get(erroredKey)).toMatchObject({
        animation: 'unavailable',
        authority: 'stale',
        observedAt: 42
      })
      expect($lunarCitySnapshot.get().sources.filter(source => source.source.startsWith('fleet:'))).toEqual([
        { authority: 'authoritative', observedAt: 60_043, source: 'fleet:local' },
        { authority: 'stale', observedAt: 42, source: 'fleet:ssh-1' },
        { authority: 'stale', error: 'degraded', observedAt: 42, source: 'fleet:ssh-2' }
      ])
    } finally {
      stop?.()
      _resetFleetRosterForTests()

      if (priorDesktop) {
        desktopWindow.hermesDesktop = priorDesktop
      } else {
        delete desktopWindow.hermesDesktop
      }
    }
  })

  it('keeps optional Kanban source failures isolated from session entities and releases the source subscription', async () => {
    const sessions = atom([
      {
        connection_id: 'source-a',
        ended_at: null,
        id: 'session-a',
        is_active: true,
        profile: 'worker'
      } as never
    ])

    const optionalStop = vi.fn()
    let optionalInvalidate!: () => void

    const optionalSource = {
      read: vi.fn(async () => ({
        authoritative: false,
        entities: [],
        sources: [
          {
            authority: 'unknown' as const,
            error: 'Kanban plugin unavailable',
            observedAt: 700,
            source: 'kanban:source-a'
          }
        ]
      })),
      start: vi.fn((listener: () => void) => {
        optionalInvalidate = listener

        return optionalStop
      })
    }

    const stop = startLunarCityReconciler({
      now: () => 700,
      sources: {
        $fleetRoster: atom(null),
        $sessions: sessions,
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        optionalSources: [optionalSource],
        refreshFleet: async () => ({ error: 'Fleet roster bridge unavailable', status: 'failed' })
      }
    })

    await flush()

    const snapshot = $lunarCitySnapshot.get()
    const sessionEntity = [...snapshot.entities.values()].find(entity => entity.identity.kind === 'session')

    expect(sessionEntity).toMatchObject({
      authority: 'stale',
      identity: { connectionId: 'source-a', profile: 'worker', sessionId: 'session-a' }
    })
    expect(snapshot.sources).toEqual([
      {
        authority: 'unknown',
        error: 'Kanban plugin unavailable',
        observedAt: 700,
        source: 'kanban:source-a'
      },
      {
        authority: 'partial',
        error: 'Registered session roster unavailable',
        observedAt: 700,
        source: 'session-registry:unavailable'
      },
      { authority: 'stale', observedAt: 0, source: 'session:source-a:worker' }
    ])

    optionalInvalidate()
    await flush()
    expect(optionalSource.read).toHaveBeenCalledTimes(2)

    stop()
    expect(optionalStop).toHaveBeenCalledOnce()
  })

  it('keeps expired cached sessions stale across unrelated reconciliation until an exact store reread', async () => {
    vi.useFakeTimers()
    let now = 100

    const sessionRow = {
      connection_id: 'local',
      ended_at: null,
      id: 'shared-session',
      is_active: true,
      profile: 'worker'
    }

    const local = { ...sessionRow } as unknown as SessionInfo
    const remote = { ...sessionRow, connection_id: 'ssh-1' } as unknown as SessionInfo
    const sessions = atom([local, remote])

    const fleet = atom({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local' as const,
          connectionLabel: 'this Mac',
          handle: '@worker-local',
          profile: 'worker'
        },
        {
          connectionId: 'ssh-1',
          connectionKind: 'ssh' as const,
          connectionLabel: 'relay',
          handle: '@worker-remote',
          profile: 'worker'
        }
      ],
      sources: [
        { connectionId: 'local', kind: 'local' as const, label: 'this Mac', reachable: true },
        { connectionId: 'ssh-1', kind: 'ssh' as const, label: 'relay', reachable: true }
      ]
    })

    let readEnabled = false
    let focus!: () => void

    const stop = startLunarCityReconciler({
      freshnessMs: 50,
      now: () => now,
      sources: {
        $fleetRoster: fleet,
        $sessions: sessions,
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onFocus: listener => {
          focus = listener

          return () => undefined
        },
        readSessionList: async connectionId => {
          if (!readEnabled) {
            throw new Error('exact list unavailable')
          }

          return { sessions: [connectionId === 'local' ? local : remote] }
        },
        refreshFleet: async () => ({ observedAt: now, status: 'refreshed' })
      }
    })

    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => entity.authority)
    ).toEqual(['stale', 'stale'])

    const staleRevision = $lunarCitySnapshot.get().revision
    sessions.set([
      { ...local, git_repo_root: '/optimistic/project', title: 'optimistic local title' },
      { ...remote, title: 'optimistic remote title' }
    ])
    await flush()
    expect($lunarCitySnapshot.get().revision).toBe(staleRevision)
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt, entity.projectId])
    ).toEqual([
      ['local', 'stale', 0, undefined],
      ['ssh-1', 'stale', 0, undefined]
    ])

    readEnabled = true
    focus()
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'authoritative', 100],
      ['ssh-1', 'authoritative', 100]
    ])

    now = 150
    await vi.advanceTimersByTimeAsync(50)
    readEnabled = false
    focus()
    await flush()

    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'stale', 100],
      ['ssh-1', 'stale', 100]
    ])

    now = 160
    readEnabled = true
    focus()
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'authoritative', 160],
      ['ssh-1', 'authoritative', 160]
    ])

    now = 170
    fleet.set({
      ...fleet.get(),
      sources: fleet.get().sources.filter(source => source.connectionId !== 'local')
    })
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'stale', 160],
      ['ssh-1', 'authoritative', 170]
    ])

    now = 180
    focus()
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority])
    ).toEqual([
      ['local', 'stale'],
      ['ssh-1', 'authoritative']
    ])

    now = 190
    readEnabled = false
    fleet.set({
      ...fleet.get(),
      sources: [
        { connectionId: 'local', kind: 'local' as const, label: 'this Mac', reachable: true },
        ...fleet.get().sources
      ]
    })
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority])
    ).toEqual([
      ['local', 'stale'],
      ['ssh-1', 'stale']
    ])

    now = 200
    readEnabled = true
    focus()
    await flush()
    expect(
      [...$lunarCitySnapshot.get().entities.values()]
        .filter(entity => entity.identity.kind === 'session')
        .map(entity => [entity.identity.connectionId, entity.authority, entity.observedAt])
    ).toEqual([
      ['local', 'authoritative', 200],
      ['ssh-1', 'authoritative', 200]
    ])
    stop()
  })

  it('lets an optional Kanban failure retain only its own stale rows while core sessions keep reconciling', async () => {
    const firstSession = {
      connection_id: 'source-a',
      ended_at: null,
      id: 'session-one',
      is_active: true,
      profile: 'worker'
    } as never

    const secondSession = {
      connection_id: 'source-a',
      ended_at: null,
      id: 'session-two',
      is_active: true,
      profile: 'worker'
    } as never

    const sessions = atom([firstSession])

    const kanbanIdentity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban' as const,
      profile: 'default',
      taskId: 'task-one'
    }

    const kanbanEntity = {
      animation: 'work',
      authority: 'authoritative' as const,
      destination: 'project' as const,
      identity: kanbanIdentity,
      key: entityKey(kanbanIdentity),
      observedAt: 900
    }

    let mode: 'authoritative' | 'unavailable' | 'empty' = 'authoritative'
    let invalidateOptional!: () => void
    let emit!: (event: unknown) => void

    const optionalSource = {
      read: async () => {
        if (mode === 'authoritative') {
          return {
            authoritative: true,
            entities: [kanbanEntity],
            sources: [{ authority: 'authoritative' as const, observedAt: 900, source: 'kanban:source-a:default' }]
          }
        }

        if (mode === 'unavailable') {
          return {
            authoritative: false,
            entities: [],
            sources: [
              {
                authority: 'unknown' as const,
                error: 'Kanban plugin unavailable',
                observedAt: 901,
                source: 'kanban:source-a:default'
              }
            ]
          }
        }

        return {
          authoritative: true,
          entities: [],
          sources: [{ authority: 'authoritative' as const, observedAt: 902, source: 'kanban:source-a:default' }]
        }
      },
      start: (listener: () => void) => {
        invalidateOptional = listener

        return () => undefined
      }
    }

    const stop = startLunarCityReconciler({
      now: () => 900,
      sources: {
        $fleetRoster: atom({
          agents: [
            {
              connectionId: 'source-a',
              connectionKind: 'local' as const,
              connectionLabel: 'source a',
              handle: '@worker-source-a',
              profile: 'worker'
            }
          ],
          sources: [{ connectionId: 'source-a', kind: 'local' as const, label: 'source a', reachable: true }]
        }),
        $sessions: sessions,
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        optionalSources: [optionalSource],
        readSessionList: async () => ({ sessions: sessions.get() }),
        refreshFleet: async () => ({ status: 'retained' })
      }
    })

    await flush()
    expect($lunarCitySnapshot.get().entities.has(entityKey(kanbanIdentity))).toBe(true)

    mode = 'unavailable'
    sessions.set([secondSession])
    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'session-two', type: 'session.changed' })
    await flush()

    const unavailableSnapshot = $lunarCitySnapshot.get()
    expect([...unavailableSnapshot.entities.values()].map(entity => entity.identity)).toContainEqual(
      expect.objectContaining({ kind: 'session', sessionId: 'session-two' })
    )
    expect([...unavailableSnapshot.entities.values()].map(entity => entity.identity)).not.toContainEqual(
      expect.objectContaining({ kind: 'session', sessionId: 'session-one' })
    )
    expect(unavailableSnapshot.entities.get(entityKey(kanbanIdentity))).toMatchObject({
      animation: 'unavailable',
      authority: 'stale',
      destination: 'unknown'
    })

    mode = 'empty'
    invalidateOptional()
    await flush()

    expect($lunarCitySnapshot.get().entities.has(entityKey(kanbanIdentity))).toBe(false)
    stop()
  })

  it('reads every registered profile owner and keeps colliding sessions and subagents connection-exact', async () => {
    const fleet = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const sessionReads = vi.fn(async (connectionId: string, profileName: string) => ({
      sessions: [
        {
          ended_at: null,
          id: 'shared-session',
          is_active: true,
          profile: profileName,
          title: `${connectionId} session`
        }
      ]
    }))

    const delegationReads = vi.fn(async (connectionId: string) => ({
      active: [
        {
          goal: `${connectionId} child`,
          owner_agent_session_id: 'shared-session',
          started_at: 1,
          status: 'running',
          subagent_id: 'shared-child',
          tool_count: 0
        }
      ]
    }))

    const stop = startLunarCityReconciler({
      now: () => 500,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        readDelegationStatus: delegationReads,
        readSessionList: sessionReads,
        refreshFleet: async () => ({ observedAt: 500, status: 'refreshed' })
      }
    })

    await flush()
    await flush()

    const identities = [...$lunarCitySnapshot.get().entities.values()].map(entity => entity.identity)
    expect(identities).toContainEqual({
      connectionId: 'source-a',
      kind: 'session',
      profile: 'worker',
      sessionId: 'shared-session'
    })
    expect(identities).toContainEqual({
      connectionId: 'source-b',
      kind: 'session',
      profile: 'worker',
      sessionId: 'shared-session'
    })
    expect(identities).toContainEqual({
      connectionId: 'source-a',
      kind: 'subagent',
      profile: 'worker',
      sessionId: 'shared-session',
      subagentId: 'shared-child'
    })
    expect(identities).toContainEqual({
      connectionId: 'source-b',
      kind: 'subagent',
      profile: 'worker',
      sessionId: 'shared-session',
      subagentId: 'shared-child'
    })
    expect(sessionReads.mock.calls).toEqual([
      ['source-a', 'worker'],
      ['source-b', 'worker']
    ])
    expect(delegationReads.mock.calls).toEqual([
      ['source-a', 'worker'],
      ['source-b', 'worker']
    ])
    stop()
  })

  it('retains a failed registered owner stale, refreshes it on reconnect, and does not hot-read healthy owners', async () => {
    const fleet = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    let emit!: (event: unknown) => void
    let remoteFails = false

    const sessionReads = vi.fn(async (connectionId: string) => {
      if (connectionId === 'source-b' && remoteFails) {
        throw new Error('remote sleeping')
      }

      return {
        sessions: [{ ended_at: null, id: 'shared-session', is_active: true, profile: 'worker' }]
      }
    })

    const stop = startLunarCityReconciler({
      now: () => 600,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readSessionList: sessionReads,
        refreshFleet: async () => ({ observedAt: 600, status: 'refreshed' })
      }
    })

    await flush()
    await flush()
    expect(sessionReads).toHaveBeenCalledTimes(2)

    remoteFails = true
    emit({ connectionId: 'source-b', profile: 'worker', session_id: 'shared-session', type: 'session.changed' })
    await flush()
    await flush()
    expect(sessionReads.mock.calls.map(call => call[0])).toEqual(['source-a', 'source-b', 'source-b'])
    expect(
      $lunarCitySnapshot
        .get()
        .entities.get(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toMatchObject({ authority: 'stale' })

    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'shared-session', type: 'session.changed' })
    await flush()
    await flush()
    expect(sessionReads.mock.calls.map(call => call[0])).toEqual(['source-a', 'source-b', 'source-b', 'source-a'])
    expect(
      $lunarCitySnapshot
        .get()
        .entities.get(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toMatchObject({ authority: 'stale' })

    remoteFails = false
    emit({ connectionId: 'source-b', profile: 'worker', type: 'gateway.ready' })
    await flush()
    await flush()
    expect(sessionReads.mock.calls.map(call => call[0])).toEqual([
      'source-a',
      'source-b',
      'source-b',
      'source-a',
      'source-b'
    ])
    expect(
      $lunarCitySnapshot
        .get()
        .entities.get(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toMatchObject({ authority: 'authoritative' })

    fleet.set({
      agents: fleet.get().agents.filter(agent => agent.connectionId === 'source-a'),
      sources: fleet.get().sources.filter(source => source.connectionId === 'source-a')
    })
    await flush()
    await flush()
    expect(
      $lunarCitySnapshot
        .get()
        .entities.get(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toMatchObject({ authority: 'stale' })
    expect(sessionReads.mock.calls.filter(call => call[0] === 'source-b')).toHaveLength(3)

    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'shared-session', type: 'session.changed' })
    await flush()
    await flush()
    expect(
      $lunarCitySnapshot
        .get()
        .entities.has(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toBe(false)
    expect($lunarCitySnapshot.get().sources.some(source => source.source === 'session:source-b:worker')).toBe(false)

    fleet.set({
      agents: [
        ...fleet.get().agents,
        {
          connectionId: 'source-b',
          connectionKind: 'local',
          connectionLabel: 'source-b',
          handle: '@worker-source-b',
          profile: 'worker'
        }
      ],
      sources: [...fleet.get().sources, { connectionId: 'source-b', kind: 'local', label: 'source-b', reachable: true }]
    })
    await flush()
    await flush()
    expect(sessionReads.mock.calls.filter(call => call[0] === 'source-b')).toHaveLength(4)
    expect(
      $lunarCitySnapshot
        .get()
        .entities.get(
          entityKey({ connectionId: 'source-b', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
        )
    ).toMatchObject({ authority: 'authoritative' })
    stop()
  })

  it('treats an authoritative empty roster as final-owner removal without cached resurrection', async () => {
    const fleet = atom<DesktopAgentRoster>({
      agents: [
        {
          connectionId: 'source-a',
          connectionKind: 'local',
          connectionLabel: 'source a',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'source-a', kind: 'local', label: 'source a', reachable: true }]
    })

    let emit!: (event: unknown) => void

    const sessionReads = vi.fn(async () => ({
      sessions: [{ ended_at: null, id: 'session-a', is_active: true, profile: 'worker' }]
    }))

    const sessionKey = entityKey({
      connectionId: 'source-a',
      kind: 'session',
      profile: 'worker',
      sessionId: 'session-a'
    })

    const stop = startLunarCityReconciler({
      now: () => 700,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readSessionList: sessionReads,
        refreshFleet: async () => ({ observedAt: 700, status: 'refreshed' })
      }
    })

    await flush()
    expect($lunarCitySnapshot.get().entities.get(sessionKey)).toMatchObject({ authority: 'authoritative' })

    const overflowAgents = [
      fleet.get().agents[0]!,
      ...Array.from({ length: 256 }, (_, index) => ({
        connectionId: `overflow-${index}`,
        connectionKind: 'local' as const,
        connectionLabel: `overflow-${index}`,
        handle: '@worker',
        profile: 'worker'
      }))
    ]

    fleet.set({
      agents: overflowAgents,
      sources: overflowAgents.map(agent => ({
        connectionId: agent.connectionId,
        kind: 'local',
        label: agent.connectionLabel,
        reachable: true
      }))
    })
    await flush()
    expect($lunarCitySnapshot.get().entities.get(sessionKey)).toMatchObject({ authority: 'stale' })
    expect($lunarCitySnapshot.get().sources).toContainEqual(
      expect.objectContaining({ authority: 'partial', source: 'session-registry:overflow' })
    )
    expect(sessionReads).toHaveBeenCalledOnce()

    fleet.set({ agents: [], sources: [] })
    await flush()
    expect($lunarCitySnapshot.get().entities.get(sessionKey)).toMatchObject({ authority: 'stale' })

    emit({ connectionId: 'source-a', profile: 'worker', type: 'gateway.ready' })
    await flush()
    expect($lunarCitySnapshot.get().entities.has(sessionKey)).toBe(false)
    expect(sessionReads).toHaveBeenCalledOnce()
    stop()
  })

  it('fails closed on an oversized roster without instantiating owner readers or vacuous authority', async () => {
    const agents = Array.from({ length: 257 }, (_, index) => ({
      connectionId: `source-${index}`,
      connectionKind: 'local' as const,
      connectionLabel: `source-${index}`,
      handle: '@worker',
      profile: 'worker'
    }))

    const fleet = atom<DesktopAgentRoster>({
      agents,
      sources: agents.map(agent => ({
        connectionId: agent.connectionId,
        kind: 'local' as const,
        label: agent.connectionLabel,
        reachable: true
      }))
    })

    const sessionReads = vi.fn()

    const stop = startLunarCityReconciler({
      now: () => 800,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        readSessionList: sessionReads,
        refreshFleet: async () => ({ observedAt: 800, status: 'refreshed' })
      }
    })

    await flush()
    expect(sessionReads).not.toHaveBeenCalled()
    expect($lunarCitySnapshot.get().entities.size).toBe(0)
    expect($lunarCitySnapshot.get().sources).toContainEqual({
      authority: 'partial',
      error: 'Registered session owner limit exceeded',
      observedAt: 800,
      source: 'session-registry:overflow'
    })
    stop()
  })

  it('downgrades retained exact profile, session, and subagent rows when a healthy roster becomes unavailable', async () => {
    const fleet = atom<DesktopAgentRoster | null>({
      agents: [
        {
          connectionId: 'source-a',
          connectionKind: 'local',
          connectionLabel: 'source a',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'source-a', kind: 'local', label: 'source a', reachable: true }]
    })

    const sessionReads = vi.fn(async () => ({
      sessions: [{ ended_at: null, id: 'parent', is_active: true, profile: 'worker' }]
    }))

    const delegationReads = vi.fn(async () => ({
      active: [
        {
          goal: 'child',
          owner_agent_session_id: 'parent',
          started_at: 1,
          status: 'running',
          subagent_id: 'child',
          tool_count: 0
        }
      ]
    }))

    const stop = startLunarCityReconciler({
      now: () => 900,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        readDelegationStatus: delegationReads,
        readSessionList: sessionReads,
        refreshFleet: async () => ({ observedAt: 900, status: 'refreshed' })
      }
    })

    await flush()

    const exactKeys = [
      entityKey({ connectionId: 'source-a', kind: 'profile', profile: 'worker' }),
      entityKey({ connectionId: 'source-a', kind: 'session', profile: 'worker', sessionId: 'parent' }),
      entityKey({
        connectionId: 'source-a',
        kind: 'subagent',
        profile: 'worker',
        sessionId: 'parent',
        subagentId: 'child'
      })
    ]

    expect(exactKeys.map(key => $lunarCitySnapshot.get().entities.get(key)?.authority)).toEqual([
      'authoritative',
      'authoritative',
      'authoritative'
    ])

    fleet.set(null)
    await flush()
    expect(exactKeys.map(key => $lunarCitySnapshot.get().entities.get(key)?.authority)).toEqual([
      'stale',
      'stale',
      'stale'
    ])
    expect(sessionReads).toHaveBeenCalledOnce()
    expect(delegationReads).toHaveBeenCalledOnce()
    expect($lunarCitySnapshot.get().sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ authority: 'stale', source: 'fleet:source-a' }),
        expect.objectContaining({ authority: 'stale', source: 'session:source-a:worker' }),
        expect.objectContaining({ authority: 'stale', source: 'delegation:source-a:worker' }),
        expect.objectContaining({ authority: 'partial', source: 'session-registry:unavailable' })
      ])
    )
    stop()
  })

  it('replaces a healthy sibling profile while retaining the failed sibling stale on one connection', async () => {
    const fleet = atom<DesktopAgentRoster>({
      agents: ['scout', 'worker'].map(profileName => ({
        connectionId: 'source-a',
        connectionKind: 'local' as const,
        connectionLabel: 'source a',
        handle: `@${profileName}`,
        profile: profileName
      })),
      sources: [{ connectionId: 'source-a', kind: 'local', label: 'source a', reachable: true }]
    })

    let emit!: (event: unknown) => void
    let scoutFails = false
    let workerRevision = 1

    const reads = vi.fn(async (_connectionId: string, profileName: string) => {
      if (profileName === 'scout' && scoutFails) {
        throw new Error('scout unavailable')
      }

      return {
        sessions: [
          {
            ended_at: null,
            id: 'shared-session',
            is_active: true,
            profile: profileName,
            title: profileName === 'worker' ? `worker-${workerRevision}` : 'scout'
          }
        ]
      }
    })

    const stop = startLunarCityReconciler({
      now: () => 700 + workerRevision,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readSessionList: reads,
        refreshFleet: async () => ({ observedAt: 700, status: 'refreshed' })
      }
    })

    await flush()
    await flush()

    scoutFails = true
    emit({ connectionId: 'source-a', profile: 'scout', session_id: 'shared-session', type: 'session.changed' })
    await flush()
    workerRevision = 2
    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'shared-session', type: 'session.changed' })
    await flush()

    const snapshot = $lunarCitySnapshot.get()
    expect(
      snapshot.entities.get(
        entityKey({ connectionId: 'source-a', kind: 'session', profile: 'scout', sessionId: 'shared-session' })
      )
    ).toMatchObject({ authority: 'stale' })
    expect(
      snapshot.entities.get(
        entityKey({ connectionId: 'source-a', kind: 'session', profile: 'worker', sessionId: 'shared-session' })
      )
    ).toMatchObject({ authority: 'authoritative', observedAt: 702 })
    expect(snapshot.sources).toContainEqual(
      expect.objectContaining({
        authority: 'stale',
        observedAt: 701,
        source: 'session:source-a:scout'
      })
    )
    expect(snapshot.sources).toContainEqual({
      authority: 'authoritative',
      observedAt: 702,
      source: 'session:source-a:worker'
    })
    stop()
  })

  it('keeps delegation failure stale independently and purges children whose exact parent disappears', async () => {
    const fleet = atom<DesktopAgentRoster>({
      agents: [
        {
          connectionId: 'source-a',
          connectionKind: 'local',
          connectionLabel: 'source a',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'source-a', kind: 'local', label: 'source a', reachable: true }]
    })

    let now = 800
    let emit!: (event: unknown) => void
    let sessionExists = true
    let delegationFails = false

    const stop = startLunarCityReconciler({
      now: () => now,
      sources: {
        $fleetRoster: fleet,
        $sessions: atom([]),
        $subagentsBySession: atom({}),
        legacySingleBackend: () => false,
        onEvent: listener => {
          emit = listener

          return () => undefined
        },
        readDelegationStatus: async () => {
          if (delegationFails) {
            throw new Error('delegation unavailable')
          }

          return {
            active: sessionExists
              ? [
                  {
                    goal: 'child',
                    owner_agent_session_id: 'parent',
                    started_at: 1,
                    status: 'running',
                    subagent_id: 'child',
                    tool_count: 0
                  }
                ]
              : []
          }
        },
        readSessionList: async () => ({
          sessions: sessionExists ? [{ ended_at: null, id: 'parent', is_active: true, profile: 'worker' }] : []
        }),
        refreshFleet: async () => ({ observedAt: now, status: 'refreshed' })
      }
    })

    await flush()
    await flush()

    const childKey = entityKey({
      connectionId: 'source-a',
      kind: 'subagent',
      profile: 'worker',
      sessionId: 'parent',
      subagentId: 'child'
    })

    expect($lunarCitySnapshot.get().entities.get(childKey)).toMatchObject({
      authority: 'authoritative',
      observedAt: 800
    })

    now = 810
    delegationFails = true
    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'parent', type: 'subagent.progress' })
    await flush()
    expect($lunarCitySnapshot.get().entities.get(childKey)).toMatchObject({ authority: 'stale', observedAt: 800 })
    expect($lunarCitySnapshot.get().sources).toContainEqual(
      expect.objectContaining({
        authority: 'stale',
        observedAt: 800,
        source: 'delegation:source-a:worker'
      })
    )

    now = 820
    sessionExists = false
    emit({ connectionId: 'source-a', profile: 'worker', session_id: 'parent', type: 'session.changed' })
    await flush()
    expect($lunarCitySnapshot.get().entities.has(childKey)).toBe(false)
    stop()
  })
})
