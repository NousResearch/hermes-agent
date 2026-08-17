import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

vi.mock('@/app/chat/session-view', async () => {
  const { atom } = await import('nanostores')

  return { PRIMARY_SESSION_VIEW: { $awaitingResponse: atom(false), $busy: atom(false) } }
})
vi.mock('@/app/open-session', () => ({ openSession: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', async () => {
  const { atom } = await import('nanostores')

  return { $narrowViewport: atom(false) }
})
vi.mock('@/contrib/events', () => ({ onGatewayEvent: vi.fn() }))
vi.mock('@/hermes', () => ({ deleteProfile: vi.fn(), getLogs: vi.fn(), getStatus: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/system-actions', () => ({ runGatewayRestart: vi.fn() }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeSessionId: atom(null),
    $connection: atom(null),
    $currentCwd: atom(''),
    $currentModel: atom(''),
    $gatewayState: atom('open'),
    $selectedStoredSessionId: atom(null)
  }
})
vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return {
    $focusedRuntimeId: atom(null),
    $focusedSessionState: atom(null),
    $focusedStoredSessionId: atom(null),
    $sessionStates: atom({})
  }
})
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  const profiles = atom([
    {
      has_env: false,
      is_default: false,
      model: null,
      name: 'cached-only',
      path: '/profiles/cached-only',
      provider: null,
      skill_count: 0
    }
  ])

  return {
    $activeGatewayProfile: atom('remote-worker'),
    $profiles: profiles,
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn(),
    newSessionInProfile: vi.fn(),
    normalizeProfileKey: (value: null | string | undefined) => (value ?? '').trim() || 'default',
    refreshProfiles: vi.fn(async () => profiles.get()),
    selectProfile: vi.fn(),
    setActiveProfile: vi.fn(),
    setShowAllProfiles: vi.fn()
  }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return {
    $gateway: atom(null),
    ensureGatewayForAgent: vi.fn(),
    openGatewayForAgent: vi.fn(),
    openGatewayForProfile: vi.fn(),
    requestGatewayForAgent: vi.fn(
      async (connectionId: string, profile: string, method: string, params: Record<string, unknown>) => ({
        connectionId,
        method,
        params,
        profile
      })
    ),
    requestGatewayForProfile: vi.fn(async (profile: string, method: string, params: Record<string, unknown>) => ({
      method,
      params,
      profile
    })),
    retireLocalProfileGateways: vi.fn()
  }
})

const { host } = await import('./index')
const { deleteProfile } = await import('@/hermes')
const { requestGatewayForAgent, requestGatewayForProfile, retireLocalProfileGateways } = await import('@/store/gateway')
const { $profiles, refreshProfiles } = await import('@/store/profile')

const profile = (name: string): ProfileInfo => ({
  has_env: false,
  is_default: name === 'default',
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

afterEach(() => {
  vi.clearAllMocks()
  $profiles.set([profile('cached-only')])
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('connection-aware plugin host APIs', () => {
  it('retires a profile gateway before deleting it', async () => {
    const order: string[] = []

    vi.mocked(retireLocalProfileGateways).mockImplementationOnce(() => {
      order.push('retire')
    })
    vi.mocked(deleteProfile).mockImplementationOnce(async () => {
      order.push('delete')

      return { ok: true, path: '/profiles/worker' }
    })

    await host.deleteProfile('worker')

    expect(order).toEqual(['retire', 'delete'])
    expect(retireLocalProfileGateways).toHaveBeenCalledWith('worker')
  })

  it('refreshes the profile inventory before asking Electron for routes', async () => {
    const getProfileRoutes = vi.fn(async () => [
      { connectionId: 'connection-local', mode: 'local', profile: 'desktop-primary', targetProfile: 'desktop-primary' },
      {
        connectionId: 'connection-remote',
        mode: 'remote',
        profile: 'remote-worker',
        targetProfile: 'backend-worker'
      }
    ])

    vi.mocked(refreshProfiles).mockResolvedValueOnce([profile('desktop-primary'), profile('remote-worker')])
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { getProfileRoutes }

    const routes = await host.profileRoutes()

    expect(routes).toEqual([
      { connectionId: 'connection-local', mode: 'local', profile: 'desktop-primary', targetProfile: 'desktop-primary' },
      {
        connectionId: 'connection-remote',
        mode: 'remote',
        profile: 'remote-worker',
        targetProfile: 'backend-worker'
      }
    ])
    expect(refreshProfiles).toHaveBeenCalledOnce()
    expect(getProfileRoutes).toHaveBeenCalledWith(['desktop-primary', 'remote-worker'])
  })

  it('falls back to the cached profile inventory when refresh fails', async () => {
    const getProfileRoutes = vi.fn(async () => [
      { connectionId: 'connection-cached', mode: 'remote', profile: 'cached-worker', targetProfile: 'cached-worker' }
    ])

    $profiles.set([profile('cached-worker')])
    vi.mocked(refreshProfiles).mockRejectedValueOnce(new Error('profile backend unavailable'))
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { getProfileRoutes }

    await expect(host.profileRoutes()).resolves.toEqual([
      { connectionId: 'connection-cached', mode: 'remote', profile: 'cached-worker', targetProfile: 'cached-worker' }
    ])
    expect(getProfileRoutes).toHaveBeenCalledWith(['cached-worker'])
  })

  it('forwards a targeted request without changing foreground state', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'remote-worker',
      targetProfile: 'backend-worker'
    }

    const result = await host.requestProfile(route, 'profiles.list', { include_sessions: true })

    expect(result).toEqual({
      connectionId: 'source-a',
      method: 'profiles.list',
      params: { include_sessions: true },
      profile: 'remote-worker'
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'remote-worker', 'profiles.list', {
      include_sessions: true
    })
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps the profile-only request overload as a legacy fallback', async () => {
    const result = await host.requestProfile('legacy-worker', 'profiles.list', { include_sessions: true })

    expect(result).toEqual({
      method: 'profiles.list',
      params: { include_sessions: true },
      profile: 'legacy-worker'
    })
    expect(requestGatewayForProfile).toHaveBeenCalledWith('legacy-worker', 'profiles.list', {
      include_sessions: true
    })
  })

  it('rejects a profile-only request when the current registry makes it ambiguous', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getAgentRoster: vi.fn(async () => ({
        agents: [
          { connectionId: 'source-a', profile: 'research' },
          { connectionId: 'source-b', profile: 'research' }
        ],
        sources: [
          { connectionId: 'source-a', kind: 'remote', label: 'A' },
          { connectionId: 'source-b', kind: 'remote', label: 'B' }
        ]
      }))
    }

    await expect(host.requestProfile('research', 'profiles.list')).rejects.toThrow(/route descriptor/i)
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps profile-only compatibility when sole-local enumeration fails', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getAgentRoster: vi.fn(async () => ({
        agents: [],
        sources: [{ connectionId: 'local', kind: 'local', label: 'This device' }]
      }))
    }

    await expect(host.requestProfile('research', 'profiles.list')).resolves.toEqual({
      method: 'profiles.list',
      params: {},
      profile: 'research'
    })
  })

  it('rejects profile-only routing when another source is undialed', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getAgentRoster: vi.fn(async () => ({
        agents: [{ connectionId: 'local', profile: 'research' }],
        sources: [
          { connectionId: 'local', kind: 'local', label: 'This device' },
          { connectionId: 'homelab', kind: 'ssh', label: 'Homelab' }
        ]
      }))
    }

    await expect(host.requestProfile('research', 'profiles.list')).rejects.toThrow(/route descriptor/i)
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })
})
