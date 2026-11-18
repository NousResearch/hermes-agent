import type * as HermesSdk from '@hermes/plugin-sdk'
import { queryClient } from '@hermes/plugin-sdk'
import { afterEach, expect, it, vi } from 'vitest'

import { $lastRoster } from './data'
import type * as BotsData from './data'
import { $lastSources } from './roster-pane-lifecycle'
import { startRosterPublisher } from './roster-publisher'

const { snapshot, ownerLabels } = vi.hoisted(() => ({
  snapshot: vi.fn(),
  ownerLabels: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: {
      ...sdk.host,
      setWorkspaceOwnerLabel: ownerLabels,
      state: { ...sdk.host.state, connectionId: sdk.atom<string | null>('local'), gateway: sdk.atom('open') }
    }
  }
})

vi.mock('./data', async importOriginal => {
  const data = await importOriginal<typeof BotsData>()

  return {
    ...data,
    rosterQueryOptions: (connectionId: null | string) => ({
      ...data.rosterQueryOptions(connectionId),
      queryFn: () => snapshot(connectionId),
      refetchInterval: false as const
    })
  }
})

vi.mock('./profile-ops', () => ({ pullServerAvatars: vi.fn() }))
vi.mock('./soul', () => ({ backfillMessagingProtocol: vi.fn() }))
vi.mock('./roster-actions', () => ({ trackInboundActivity: vi.fn() }))

let stop = () => {}

afterEach(() => {
  stop()
  queryClient.clear()
  $lastRoster.set([])
  $lastSources.set([])
  vi.clearAllMocks()
})

it('publishes the roster with no Bots pane mounted, without its ghosts, and stops on demand', async () => {
  const sources = [{ connectionId: 'local', kind: 'local', label: 'This Mac', ok: true }]

  snapshot.mockResolvedValue({
    fetchedAt: 1,
    profiles: [{ name: 'ada' }, { name: 'gone', ghost: true }],
    sources
  })

  stop = startRosterPublisher()

  await vi.waitFor(() => expect($lastRoster.get().map(row => row.name)).toEqual(['ada']))
  expect($lastSources.get()).toEqual(sources)
  expect(ownerLabels).toHaveBeenCalledWith(expect.any(String), expect.stringMatching(/ada/i))

  stop()
  snapshot.mockResolvedValue({ fetchedAt: 2, profiles: [{ name: 'later' }], sources })
  await queryClient.refetchQueries()
  expect($lastRoster.get().map(row => row.name)).toEqual(['ada'])
})
