/**
 * The roster snapshot publish (`usePublishRosterSnapshot`) is the one place the
 * 5 s roster poll fans out into shared state: `$lastRoster`, workspace owner
 * labels, server-meta merge, avatar pulls, inbound-activity tracking. Every poll
 * stamps a fresh `fetchedAt`, so the query envelope is a new object each tick
 * even when the roster did not change; the publish must key on the
 * structurally-shared `profiles` / `sources` subtrees so an identical roster is
 * NOT republished (every subscriber re-rendered and every side effect re-ran
 * every 5 s), while a changed roster still is.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, mergeServerMeta, trackInboundActivity } = vi.hoisted(() => ({
  hostMock: {
    agents: vi.fn(),
    request: vi.fn(),
    setWorkspaceOwnerLabel: vi.fn(),
    state: { connectionId: { get: vi.fn(() => 'local') }, profile: { get: () => 'default' } }
  },
  mergeServerMeta: vi.fn(),
  trackInboundActivity: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { useQuery } = await import('@tanstack/react-query')

  return {
    atom,
    host: hostMock,
    queryClient: { getQueryData: vi.fn(), invalidateQueries: vi.fn() },
    useQuery,
    useValue: (store: { get: () => unknown }) => store.get()
  }
})
vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))
vi.mock('./profile-ops', () => ({ mergeServerMeta, pullServerAvatars: vi.fn() }))
vi.mock('./roster-actions', () => ({ trackInboundActivity }))
vi.mock('./soul', () => ({ backfillMessagingProtocol: vi.fn() }))

import { $lastRoster, useRoster } from './data'
import { $rosterOrder, persistRosterOrder, rosterSectionScope } from './roster-order'
import { usePublishRosterSnapshot } from './roster-pane-lifecycle'

const rows: RosterRow[] = [{ name: 'default' }, { name: 'coder', last_session: { preview: 'hi' } }]

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

/** The BotsPane wiring: the real roster query feeding the real publish hook. */
function renderPane() {
  return renderHook(
    () => {
      const query = useRoster()
      const live = Array.isArray(query.data?.profiles) ? query.data.profiles : null
      const roster = live ?? []

      usePublishRosterSnapshot({ data: query.data, live, roster, allMeta: {}, activeSourceRoster: roster })

      return query
    },
    { wrapper: wrapper() }
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  $lastRoster.set([])
  $rosterOrder.set({})
  hostMock.agents.mockRejectedValue(new Error('no union roster'))
})

describe('usePublishRosterSnapshot', () => {
  it('does not republish an unchanged roster when only the poll envelope changed', async () => {
    hostMock.request.mockResolvedValue({ profiles: rows })

    const { result } = renderPane()

    await waitFor(() => expect(result.current.data?.profiles).toHaveLength(2))
    const firstProfiles = result.current.data?.profiles
    const firstFetchedAt = result.current.data?.fetchedAt
    const published = $lastRoster.get()
    expect(published.map(row => row.name)).toEqual(['default', 'coder'])
    expect(trackInboundActivity).toHaveBeenCalledTimes(1)

    // Second poll: same rows from the gateway, new issue stamp on the envelope.
    await new Promise(resolve => setTimeout(resolve, 5))
    await result.current.refetch()
    await waitFor(() => expect(result.current.data?.fetchedAt).not.toBe(firstFetchedAt))

    // Structural sharing keeps the unchanged subtree reference-stable...
    expect(result.current.data?.profiles).toBe(firstProfiles)
    // ...so the publish does not fan out again.
    expect($lastRoster.get()).toBe(published)
    expect(trackInboundActivity).toHaveBeenCalledTimes(1)
  })

  it('republishes when a row changed', async () => {
    hostMock.request.mockResolvedValue({ profiles: rows })

    const { result } = renderPane()

    await waitFor(() => expect(result.current.data?.profiles).toHaveLength(2))
    const published = $lastRoster.get()

    hostMock.request.mockResolvedValue({ profiles: [rows[0], { ...rows[1], last_session: { preview: 'later' } }] })
    await result.current.refetch()

    await waitFor(() => expect($lastRoster.get()).not.toBe(published))
    expect($lastRoster.get()[1]?.last_session?.preview).toBe('later')
    expect(trackInboundActivity).toHaveBeenCalledTimes(2)
  })

  it('skips pruning custom order when host.agents transiently fails, then prunes once sources arrive', async () => {
    const localScope = rosterSectionScope('sec-a', 'local')
    const sshScope = rosterSectionScope('sec-a', 'ssh-box')

    persistRosterOrder({
      [localScope]: ['local::default', 'local::deleted'],
      [sshScope]: ['ssh-box::remote-bot']
    })

    // First poll: host.agents() rejects -> degraded snapshot has no data.sources;
    // custom order must stay untouched rather than wiping local::* and ssh-box::*.
    hostMock.request.mockResolvedValue({ profiles: rows })
    const { result } = renderPane()

    await waitFor(() => expect(result.current.data?.profiles).toHaveLength(2))
    expect($rosterOrder.get()).toEqual({
      [localScope]: ['local::default', 'local::deleted'],
      [sshScope]: ['ssh-box::remote-bot']
    })

    // Second poll: host.agents() succeeds with authoritative sources -> local::deleted
    // is pruned while ssh-box (connect-on-demand) is preserved.
    hostMock.agents.mockResolvedValueOnce({
      agents: [
        { connectionId: 'local', profile: 'default' },
        { connectionId: 'local', profile: 'coder' }
      ],
      sources: [
        { connectionId: 'local', reachable: true },
        { connectionId: 'ssh-box', error: 'connect-on-demand', reachable: false }
      ]
    })

    await result.current.refetch()
    await waitFor(() => expect(result.current.data?.sources).toHaveLength(2))

    expect($rosterOrder.get()).toEqual({
      [localScope]: ['local::default'],
      [sshScope]: ['ssh-box::remote-bot']
    })
  })
})
