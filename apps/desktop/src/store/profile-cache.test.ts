import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { buildRestGroups } from '@/app/chat/sidebar/fleet-rail'
import type { DesktopAgentRoster, DesktopRegistryConnection, HermesConnection } from '@/global'
import { $fleetRoster, _resetFleetRosterForTests, refreshFleetRoster } from '@/store/fleet-roster'
import type { ProfileInfo } from '@/types/hermes'

vi.mock('@/store/gateway', () => ({ $gateway: atom(null) }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { $profiles, $profilesByConnection, invalidateProfileListFetches, refreshActiveProfile, refreshProfiles } =
  await import('./profile')

const { $connection } = await import('./session')

const profile = (name: string): ProfileInfo => ({
  name,
  is_default: name === 'default',
  path: '',
  has_env: false,
  model: null,
  provider: null,
  skill_count: 0
})

const descriptor = (connectionId: string): HermesConnection =>
  ({
    connectionId,
    baseUrl: `https://${connectionId}.example.com`,
    mode: 'remote',
    profile: 'default'
  }) as HermesConnection

function activate(connectionId: string) {
  setApiRequestConnection(connectionId)
  $connection.set(descriptor(connectionId))
}

afterEach(() => {
  _resetFleetRosterForTests()
  invalidateProfileListFetches()
  $connection.set(null)
  $profilesByConnection.set(new Map())
  $profiles.set([])
  setApiRequestConnection(null)
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('keeps failed incoming profile reads isolated while retaining the outgoing connection cache', async () => {
  const outgoing = [profile('default'), profile('writer')]
  const api = vi.fn(async () => ({ profiles: outgoing }))
  vi.stubGlobal('window', { hermesDesktop: { api } })
  activate('source-a')
  await refreshProfiles()

  // Same profile name on both machines: profile-only invalidation cannot help.
  invalidateProfileListFetches()
  activate('source-b')
  api.mockRejectedValue(new Error('HTTP 401 {"reason":"no_cookie","login_url":"/login"}'))
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.useFakeTimers()
  const refresh = refreshActiveProfile()
  await vi.runAllTimersAsync()
  await refresh

  expect($profiles.get()).not.toContain(outgoing[1])
  activate('source-a')
  expect($profiles.get()).toBe(outgoing)
})

it('lets only a subsequent successful roster supersede cached rail names', async () => {
  const connections: DesktopRegistryConnection[] = ['source-a', 'source-b'].map(id => ({
    id,
    kind: 'remote',
    label: id,
    tokenSet: false,
    tokenPreview: null
  }))

  const roster = (names: string[], reachable = true): DesktopAgentRoster => ({
    agents: names.map(name => ({
      connectionId: 'source-a',
      connectionKind: 'remote',
      connectionLabel: 'source-a',
      profile: name,
      handle: `${name}-source-a`
    })),
    sources: connections.map(connection => ({
      connectionId: connection.id,
      kind: connection.kind,
      label: connection.label,
      reachable: connection.id === 'source-a' ? reachable : true
    }))
  })

  const restNames = () =>
    buildRestGroups({
      activeConnectionId: $connection.get()?.connectionId ?? null,
      connections,
      roster: $fleetRoster.get(),
      profilesByConnection: $profilesByConnection.get()
    })
      .find(group => group.connectionId === 'source-a')!
      .named.map(agent => agent.profile)

  const outgoing = [profile('default'), profile('old')]
  const incoming = [profile('default'), profile('builder')]
  const api = vi.fn().mockResolvedValue({ profiles: outgoing })
  const getAgentRoster = vi.fn().mockResolvedValue(roster([]))
  vi.stubGlobal('window', { hermesDesktop: { api, getAgentRoster } })
  await refreshFleetRoster({ force: true })
  activate('source-a')
  await refreshProfiles()
  activate('source-b')
  api.mockResolvedValue({ profiles: incoming })
  await refreshProfiles()
  expect(restNames()).toEqual(['old']) // The older roster must not hide a newly discovered profile.

  getAgentRoster.mockRejectedValueOnce(new Error('enumeration failed'))
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  await refreshFleetRoster({ force: true })
  expect(restNames()).toEqual(['old'])
  getAgentRoster.mockResolvedValue(roster([], false))
  await refreshFleetRoster({ force: true })
  expect(restNames()).toEqual(['old']) // A partial roster failure is not a deletion.
  expect($profiles.get()).toBe(incoming)

  for (const error of ['HTTP 401', 'connect-on-demand']) {
    // The main-process registry restores cached names but retains the probe error.
    const cachedFailure = roster(['stale'], true)
    cachedFailure.sources[0].error = error
    getAgentRoster.mockResolvedValue(cachedFailure)
    await refreshFleetRoster({ force: true })
    expect(restNames()).toEqual(['old']) // Cached names are reachable, not freshly enumerated.
  }

  for (const names of [['renamed'], ['created', 'renamed'], []]) {
    getAgentRoster.mockResolvedValue(roster(names))
    await refreshFleetRoster({ force: true })
    expect(restNames()).toEqual(names)
    expect($profiles.get()).toBe(incoming)
  }

  // A roster refresh while this source is active must not clear its foreground
  // list, but an older HTTP response must not resurrect its invalidated cache.
  activate('source-a')
  api.mockResolvedValue({ profiles: outgoing })
  await refreshProfiles()
  let resolve!: (value: { profiles: ProfileInfo[] }) => void
  api.mockReturnValueOnce(
    new Promise(done => {
      resolve = done
    })
  )
  const stale = refreshProfiles()
  getAgentRoster.mockResolvedValue(roster(['latest']))
  await refreshFleetRoster({ force: true })
  expect($profiles.get()).toBe(outgoing)
  resolve({ profiles: outgoing })
  await stale
  activate('source-b')
  expect(restNames()).toEqual(['latest'])
})

it('strands a retry during a same-profile source change without retargeting it to the incoming source', async () => {
  vi.useFakeTimers()
  const incoming = [profile('default'), profile('builder')]
  const unavailable = new Error('HTTP 503')
  const api = vi.fn().mockRejectedValueOnce(unavailable).mockResolvedValue({ profiles: incoming })
  vi.stubGlobal('window', { hermesDesktop: { api } })
  activate('source-a')
  const old = refreshProfiles().catch(error => error)
  await vi.advanceTimersByTimeAsync(0) // A is in backoff, not awaiting HTTP.

  activate('source-b') // Direct activation: no beginGatewaySwitch invalidator.
  await refreshProfiles()
  await vi.runAllTimersAsync()

  expect(await old).toBe(unavailable)
  expect(api.mock.calls.map(([request]) => request.connectionId)).toEqual(['source-a', 'source-b'])
  expect($profiles.get()).toBe(incoming)
  expect($profilesByConnection.get().get('source-b')).toBe(incoming)
})
