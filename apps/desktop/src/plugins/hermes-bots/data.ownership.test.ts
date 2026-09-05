import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, useQueryMock } = vi.hoisted(() => ({
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    agents: undefined as undefined | ReturnType<typeof vi.fn>,
    state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
  },
  useQueryMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: hostMock,
    queryClient: {},
    useQuery: useQueryMock,
    useValue: vi.fn(() => 'local')
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

const { reconcileBotModeOwnership, resetBotModeOwnershipReconciliationForTests, useRoster } = await import('./data')

const remote = (connectionId: string): RosterRow =>
  ({ connectionId, name: 'default', remoteSource: true, sourceScoped: true }) as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.agents = undefined
  useQueryMock.mockImplementation(options => options)
  resetBotModeOwnershipReconciliationForTests()
})

describe('Bot Mode ownership reconciliation', () => {
  it('repairs an absent marker once and requires readback', async () => {
    hostMock.requestProfile
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: { another: { kept: true } } }] })
      .mockResolvedValueOnce({ applied: { ui_meta: true } })
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: { 'hermes-bots': {} } }] })

    await reconcileBotModeOwnership([remote('spark01')], 10_000)
    await reconcileBotModeOwnership([remote('spark01')], 10_001)

    expect(hostMock.requestProfile).toHaveBeenCalledTimes(3)
    expect(hostMock.requestProfile).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ connectionId: 'spark01' }),
      'profiles.configure',
      { name: 'default', ui_meta: { 'hermes-bots': {} } }
    )
  })

  it('does not write when any profile already owns Bot Mode', async () => {
    hostMock.requestProfile.mockResolvedValue({
      profiles: [{ name: 'auditor', ui_meta: { 'hermes-bots': { title: 'Auditor' } } }]
    })

    await reconcileBotModeOwnership([remote('spark01')], 20_000)

    expect(hostMock.requestProfile).toHaveBeenCalledTimes(1)
    expect(hostMock.requestProfile).toHaveBeenCalledWith(expect.anything(), 'profiles.list', {})
  })

  it('retries rejected or unconfirmed repairs and keeps gateways independent', async () => {
    hostMock.requestProfile.mockImplementation(async (route, method) => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'default', ui_meta: {} }] }
      }

      return { applied: { ui_meta: route.connectionId === 'spark02' } }
    })

    await reconcileBotModeOwnership([remote('spark01'), remote('spark02')], 30_000)
    await reconcileBotModeOwnership([remote('spark01'), remote('spark02')], 30_001)

    const writes = hostMock.requestProfile.mock.calls.filter(([, method]) => method === 'profiles.configure')
    expect(writes.map(([route]) => route.connectionId)).toEqual(['spark01', 'spark02', 'spark01', 'spark02'])
  })

  it('repairs the active gateway when multi-source discovery is unavailable', async () => {
    hostMock.request
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: {} }] })
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: {} }] })
      .mockResolvedValueOnce({ applied: { ui_meta: true } })
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: { 'hermes-bots': {} } }] })

    const query = useRoster() as unknown as { queryFn: () => Promise<unknown> }
    await query.queryFn()

    expect(hostMock.request).toHaveBeenNthCalledWith(3, 'profiles.configure', {
      name: 'default',
      ui_meta: { 'hermes-bots': {} }
    })
  })

  it('repairs the active gateway after union-roster discovery rejects', async () => {
    hostMock.agents = vi.fn().mockRejectedValue(new Error('roster unavailable'))
    hostMock.request
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: {} }] })
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: {} }] })
      .mockResolvedValueOnce({ applied: { ui_meta: true } })
      .mockResolvedValueOnce({ profiles: [{ name: 'default', ui_meta: { 'hermes-bots': {} } }] })

    const query = useRoster() as unknown as { queryFn: () => Promise<unknown> }
    await query.queryFn()

    expect(hostMock.agents).toHaveBeenCalledTimes(1)
    expect(hostMock.request).toHaveBeenNthCalledWith(3, 'profiles.configure', {
      name: 'default',
      ui_meta: { 'hermes-bots': {} }
    })
  })
})
