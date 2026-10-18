import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'

const request = vi.fn()

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => 'local',
  activeGatewayProfileKey: () => 'default',
  requestGatewayForAgent: (...args: unknown[]) => request(...args)
}))

vi.mock('@/api/profiles', () => ({ getProfiles: vi.fn(async () => ({ profiles: [] })) }))

const { $pluginFleet, _resetPluginFleetForTests, scanPluginFleet, updateFleetInstall } =
  await import('./plugin-fleet-updates')

const outdatedRow = (dir: string) => ({
  name: 'claude-sub',
  key: 'claude-sub',
  version: '0.3.0',
  description: '',
  source: 'user',
  status: 'enabled',
  catalog_name: 'claude-subscription-directsdk',
  catalog_sha: 'b'.repeat(40),
  catalog_version: '0.3.3',
  installed_sha: 'a'.repeat(40),
  update_available: true,
  install_dir: dir
})

function roster(partial: Partial<DesktopAgentRoster>): DesktopAgentRoster {
  return { agents: [], sources: [], ...partial }
}

const agent = (connectionId: string, profile: string, label = connectionId) => ({
  connectionId,
  connectionKind: 'remote' as const,
  connectionLabel: label,
  profile,
  handle: `@${profile}`
})

beforeEach(() => {
  _resetPluginFleetForTests()
  request.mockReset()
})

afterEach(() => {
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

function withRoster(value: DesktopAgentRoster) {
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getAgentRoster: async () => value }
}

describe('scanPluginFleet', () => {
  it('routes every profile of a gateway through one socket, scoped by profile', async () => {
    withRoster(
      roster({
        agents: [agent('local', 'default'), agent('local', 'coder'), agent('box', 'default'), agent('box', 'work')],
        sources: [
          { connectionId: 'local', label: 'This device', kind: 'local', reachable: true },
          { connectionId: 'box', label: 'Box', kind: 'remote', reachable: true }
        ]
      })
    )
    request.mockImplementation(async (connectionId, _via, _m, params) => ({
      plugins: [outdatedRow(`/${connectionId}/${params.profile}/plugins/claude-sub`)]
    }))

    const scan = await scanPluginFleet({ force: true })

    expect(scan?.outdated.map(i => `${i.target.connectionId}/${i.target.profile}`)).toEqual([
      'local/default',
      'local/coder',
      'box/default',
      'box/work'
    ])
    // Active gateway rides the window profile; the other dials its default.
    expect(request.mock.calls.map(call => [call[0], call[1], call[3].profile])).toEqual([
      ['local', 'default', 'default'],
      ['local', 'default', 'coder'],
      ['box', 'default', 'default'],
      ['box', 'default', 'work']
    ])
  })

  it('never dials an undialed (connect-on-demand) source', async () => {
    withRoster(
      roster({
        agents: [agent('local', 'default'), agent('ssh-box', 'default', 'Build box')],
        sources: [
          { connectionId: 'local', label: 'This device', kind: 'local', reachable: true },
          { connectionId: 'ssh-box', label: 'Build box', kind: 'ssh', reachable: true, error: 'connect-on-demand' }
        ]
      })
    )
    request.mockResolvedValue({ plugins: [] })

    const scan = await scanPluginFleet({ force: true })

    expect(request.mock.calls.every(call => call[0] !== 'ssh-box')).toBe(true)
    expect(scan?.skipped).toEqual([{ connectionId: 'ssh-box', label: 'Build box', reason: 'idle' }])
  })

  it('does not dial an unreachable source that still lists cached profiles', async () => {
    withRoster(
      roster({
        agents: [agent('local', 'default'), agent('box', 'default', 'Box'), agent('box', 'work', 'Box')],
        sources: [
          { connectionId: 'local', label: 'This device', kind: 'local', reachable: true },
          { connectionId: 'box', label: 'Box', kind: 'remote', reachable: false, error: 'roster enumeration timed out' }
        ]
      })
    )
    request.mockResolvedValue({ plugins: [outdatedRow('/local/default/plugins/claude-sub')] })

    const scan = await scanPluginFleet({ force: true })

    expect(request.mock.calls.map(call => call[0])).toEqual(['local'])
    expect(scan?.outdated).toHaveLength(1)
    expect(scan?.skipped).toEqual([{ connectionId: 'box', label: 'Box', reason: 'offline' }])
  })

  it('reports an unreachable gateway and a single unreadable profile instead of dropping them', async () => {
    withRoster(
      roster({
        agents: [agent('local', 'default'), agent('local', 'studio'), agent('box', 'default', 'Box')],
        sources: [
          { connectionId: 'local', label: 'This device', kind: 'local', reachable: true },
          { connectionId: 'box', label: 'Box', kind: 'remote', reachable: true }
        ]
      })
    )
    request.mockImplementation(async (connectionId, _via, _m, params) => {
      if (connectionId === 'box' || params.profile === 'studio') {
        throw new Error('timed out')
      }

      return { plugins: [outdatedRow('/local/default/plugins/claude-sub')] }
    })

    const scan = await scanPluginFleet({ force: true })

    expect(scan?.outdated).toHaveLength(1)
    expect(scan?.skipped).toEqual(
      expect.arrayContaining([
        { connectionId: 'box', label: 'Box', reason: 'offline' },
        { connectionId: 'local', label: 'This device · studio', reason: 'unreadable' }
      ])
    )
  })

  it('refuses to fan out on a gateway that ignores the profile param (same install folder)', async () => {
    withRoster(
      roster({
        agents: [agent('old', 'default', 'Old'), agent('old', 'work', 'Old')],
        sources: [{ connectionId: 'old', label: 'Old', kind: 'remote', reachable: true }]
      })
    )
    request.mockResolvedValue({ plugins: [outdatedRow('/old/plugins/claude-sub')] })

    const scan = await scanPluginFleet({ force: true })

    expect(scan?.outdated).toEqual([])
    expect(scan?.skipped.map(entry => entry.reason)).toEqual(['unreadable', 'unreadable'])
    expect($pluginFleet.get()?.outdated).toEqual([])
  })
})

describe('updateFleetInstall', () => {
  const install = {
    target: {
      connectionId: 'box',
      connectionLabel: 'Box',
      profile: 'work',
      targetProfile: 'work',
      current: false,
      via: 'default'
    },
    row: outdatedRow('/box/work/plugins/claude-sub') as never
  }

  it('updates on the owning gateway with the profile scope and a long deadline', async () => {
    request.mockResolvedValue({ ok: true, unchanged: false })

    await expect(updateFleetInstall(install)).resolves.toEqual({ kind: 'applied' })
    expect(request).toHaveBeenCalledWith(
      'box',
      'default',
      'plugins.manage',
      { action: 'update', name: 'claude-sub', profile: 'work' },
      120_000,
      undefined,
      { spawnPriority: 'foreground' }
    )
  })

  it('surfaces a consent request and retries only with explicit acceptance', async () => {
    request.mockResolvedValueOnce({ ok: false, consent_required: true, sha: 'c'.repeat(40), delta_lines: ['+ tool x'] })

    await expect(updateFleetInstall(install)).resolves.toEqual({
      kind: 'consent',
      sha: 'cccccccc',
      deltaLines: ['+ tool x']
    })

    request.mockResolvedValueOnce({ ok: true })
    await updateFleetInstall(install, true)
    expect(request.mock.calls[1][3]).toMatchObject({ accept_capabilities: true })
  })

  it('turns a transport error into a failed row, not a throw', async () => {
    request.mockRejectedValue(new Error('socket closed'))

    await expect(updateFleetInstall(install)).resolves.toEqual({ kind: 'failed', error: 'socket closed' })
  })
})
