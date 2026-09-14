import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    activeConnectionId: vi.fn(() => 'local'),
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: vi.fn(() => 'local') } }
  }
}))

vi.mock('@hermes/plugin-sdk', () => ({ host: hostMock }))

const { requestForBot } = await import('./routing')

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.activeConnectionId.mockReturnValue('local')
  hostMock.state.connectionId.get.mockReturnValue('local')
})

describe('passive Bot asset routing', () => {
  it('reads an avatar through the already-active source without activating the Bot profile', async () => {
    hostMock.request.mockResolvedValue({ found: true, data: 'data:image/png;base64,avatar' })

    const bot = {
      connectionId: 'local',
      connectionKind: 'local',
      name: 'designer',
      route: {
        connectionId: 'local',
        mode: 'local',
        profile: 'designer',
        targetProfile: 'designer-backend'
      },
      sourceScoped: true,
      targetProfile: 'designer-backend'
    } as RosterRow

    await expect(
      requestForBot(bot, 'profiles.get_asset', { name: 'designer', asset: 'avatar' })
    ).resolves.toMatchObject({ found: true })

    expect(hostMock.request).toHaveBeenCalledWith('profiles.get_asset', {
      name: 'designer-backend',
      asset: 'avatar'
    })
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('defers an inactive source instead of dialing its dormant Bot profile', async () => {
    hostMock.state.connectionId.get.mockReturnValue('source-a')
    hostMock.activeConnectionId.mockReturnValue('source-a')

    const bot = {
      connectionId: 'source-b',
      connectionKind: 'remote',
      name: 'researcher',
      remoteSource: true,
      route: {
        connectionId: 'source-b',
        mode: 'remote',
        profile: 'researcher',
        targetProfile: 'backend-researcher'
      },
      sourceScoped: true,
      targetProfile: 'backend-researcher'
    } as RosterRow

    await expect(
      requestForBot(bot, 'profiles.get_asset', { name: 'researcher', asset: 'avatar' })
    ).rejects.toThrow(/deferred until source source-b is active/)

    expect(hostMock.request).not.toHaveBeenCalled()
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('keeps non-asset Bot RPCs on strict per-profile routing', async () => {
    hostMock.requestProfile.mockResolvedValue({ ok: true })

    const bot = {
      connectionId: 'local',
      connectionKind: 'local',
      name: 'designer',
      route: {
        connectionId: 'local',
        mode: 'local',
        profile: 'designer',
        targetProfile: 'designer-backend'
      },
      sourceScoped: true,
      targetProfile: 'designer-backend'
    } as RosterRow

    await requestForBot(bot, 'profiles.configure', { name: 'designer', soul: '# hi' })

    expect(hostMock.requestProfile).toHaveBeenCalledWith(bot.route, 'profiles.configure', {
      name: 'designer-backend',
      soul: '# hi'
    })
    expect(hostMock.request).not.toHaveBeenCalled()
  })
})
