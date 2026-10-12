import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { DesktopAgentRoster, HermesConnection } from '@/global'
import { $fleetRoster, _resetFleetRosterForTests } from '@/store/fleet-roster'
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

it('lets a fresh active-source list land even when the fleet roster arrives first', async () => {
  const list = [profile('default'), profile('writer')]
  let resolve!: (value: { profiles: ProfileInfo[] }) => void
  const api = vi.fn(() => new Promise<{ profiles: ProfileInfo[] }>(done => (resolve = done)))
  vi.stubGlobal('window', { hermesDesktop: { api } })
  activate('source-a')

  const flight = refreshProfiles()

  // After a switch the registry change forces a roster refresh that races
  // the active source's own list read; both are reads of the same backend.
  const roster: DesktopAgentRoster = {
    agents: [],
    sources: [{ connectionId: 'source-a', kind: 'remote', label: 'source-a', reachable: true }]
  }

  $fleetRoster.set(roster)
  resolve({ profiles: list })
  await flight

  expect($profiles.get()).toBe(list)
  expect($profilesByConnection.get().get('source-a')).toBe(list)
})

it('treats a null descriptor as a reconnect blip, not a source change', async () => {
  const list = [profile('default'), profile('writer')]
  const api = vi.fn(async () => ({ profiles: list }))
  vi.stubGlobal('window', { hermesDesktop: { api } })
  activate('source-a')
  await refreshProfiles()

  let resolve!: (value: { profiles: ProfileInfo[] }) => void
  api.mockReturnValueOnce(new Promise(done => (resolve = done)))
  const flight = refreshProfiles()
  $connection.set(null) // failed reconnect attempt publishes null
  expect($profiles.get()).toBe(list)
  const refreshed = [profile('default'), profile('writer'), profile('editor')]
  resolve({ profiles: refreshed })
  await flight

  expect($profiles.get()).toBe(refreshed)
  activate('source-a')
  expect($profiles.get()).toBe(refreshed)
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
