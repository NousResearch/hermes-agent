import { describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    atom: nanoAtom,
    host: { request: vi.fn(), requestProfile: vi.fn(), state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } } },
    queryClient: undefined,
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

import { mergeRemoteRosterSessionFields } from './data'

describe('remote roster session hydration', () => {
  it('carries a remote Bot Chat summary onto its thin union row', () => {
    const remote = {
      canonical_session: { id: 'olympus-chat', last_active: 100, preview: 'Current Olympus reply' },
      last_session: { last_active: 100, preview: 'Current Olympus reply' },
      name: 'default',
      worker_session: { last_active: 100 },
    } as RosterRow

    const unionRow = {
      connectionId: 'olympus',
      name: 'olympus',
      remoteSource: true,
      route: { connectionId: 'olympus', mode: 'remote', profile: 'olympus', targetProfile: 'default' },
      sourceReachable: true,
      sourceScoped: true,
      targetProfile: 'default',
    } as RosterRow

    const hydrated = mergeRemoteRosterSessionFields([unionRow], new Map([['olympus', { profiles: [remote] }]]))

    expect(hydrated[0]?.canonical_session).toEqual(remote.canonical_session)
    expect(hydrated[0]?.last_session).toEqual(remote.last_session)
    expect(hydrated[0]?.worker_session).toEqual(remote.worker_session)
  })

  it('does not erase a cached session when an optional remote field is undefined', () => {
    const existing = {
      canonical_session: { id: 'cached-chat', last_active: 50 },
      connectionId: 'olympus',
      name: 'olympus',
      remoteSource: true,
      route: { connectionId: 'olympus', mode: 'remote', profile: 'olympus', targetProfile: 'default' },
      targetProfile: 'default',
    } as RosterRow

    const remote = { canonical_session: undefined, name: 'default' } as RosterRow

    const hydrated = mergeRemoteRosterSessionFields([existing], new Map([['olympus', { profiles: [remote] }]]))

    expect(hydrated[0]?.canonical_session).toEqual(existing.canonical_session)
  })
})
