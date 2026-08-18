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
    $messages: atom([]),
    $selectedStoredSessionId: atom(null),
    requestSessionResume: vi.fn(),
    setResumeExhaustedSessionId: vi.fn()
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
    $gatewaySwapTarget: atom(null),
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
const { openSession: openSessionCore } = await import('@/app/open-session')
const { deleteProfile } = await import('@/hermes')
const { requestGatewayForAgent, requestGatewayForProfile, retireLocalProfileGateways } = await import('@/store/gateway')

const { $activeGatewayProfile, $gatewaySwapTarget, $profiles, ensureGatewayProfile, refreshProfiles } =
  await import('@/store/profile')

const { $focusedRuntimeId, $focusedSessionState, $focusedStoredSessionId } = await import('@/store/session-states')

const { $activeSessionId, $messages, $selectedStoredSessionId, requestSessionResume, setResumeExhaustedSessionId } =
  await import('@/store/session')

const setMockAtom = <T>(store: unknown, value: T) => (store as { set(next: T): void }).set(value)

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
  $activeGatewayProfile.set('remote-worker')
  $gatewaySwapTarget.set(null)
  setMockAtom($focusedRuntimeId, null)
  setMockAtom($focusedStoredSessionId, null)
  setMockAtom($focusedSessionState, null)
  setMockAtom($activeSessionId, null)
  setMockAtom($selectedStoredSessionId, null)
  setMockAtom($messages, [])
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
    // The rail paints from $profiles; skipping the refresh leaves a stale
    // badge whose click hot-loops against the deletion guard (#88769).
    expect(refreshProfiles).toHaveBeenCalled()
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

describe('profile-aware plugin session opens', () => {
  it('waits until the target Bot Chat runtime and history are on main before resolving', async () => {
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async (target: null | string | undefined) => {
      $activeGatewayProfile.set(target || 'default')
    })

    let resolved = false

    const opening = host
      .openSession('bot-chat', {
        profile: 'hyoseob',
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1_000
      })
      .then(() => {
        resolved = true
      })

    await Promise.resolve()
    await Promise.resolve()

    expect(openSessionCore).toHaveBeenCalledWith('bot-chat', expect.any(Function), 'in-place')
    expect(resolved).toBe(false)
    expect($gatewaySwapTarget.get()).toBe('hyoseob')

    setMockAtom($focusedStoredSessionId, 'bot-chat')
    setMockAtom($focusedRuntimeId, 'runtime-tile')
    setMockAtom($focusedSessionState, {
      messages: [{ id: 'tile-history', parts: [], role: 'assistant' }],
      storedSessionId: 'bot-chat'
    } as never)

    await Promise.resolve()
    expect(resolved).toBe(false)

    setMockAtom($selectedStoredSessionId, 'bot-chat')
    setMockAtom($activeSessionId, 'runtime-hyoseob')
    setMockAtom($messages, [])

    await Promise.resolve()
    expect(resolved).toBe(false)

    setMockAtom($messages, [{ id: 'history-1', parts: [], role: 'user' }] as never)

    await opening
    expect(resolved).toBe(true)
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('requests an explicit resume on a cold open where selection has not settled (#89206)', async () => {
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async (target: null | string | undefined) => {
      $activeGatewayProfile.set(target || 'default')
    })

    // Cold-start shape from the field: the persisted route already points at
    // the bot's stored session, but no selection, no runtime, no transcript.
    setMockAtom($selectedStoredSessionId, null)
    setMockAtom($activeSessionId, null)
    setMockAtom($messages, [])

    const opening = host.openSession('cold-bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000
    })

    await Promise.resolve()
    await Promise.resolve()

    // The old pre-open "already selected" precondition skipped this request,
    // stranding the pane blank until timeout. It must fire now.
    expect(requestSessionResume).toHaveBeenCalledWith('cold-bot-chat')

    setMockAtom($selectedStoredSessionId, 'cold-bot-chat')
    setMockAtom($activeSessionId, 'runtime-cold')
    setMockAtom($messages, [{ id: 'history-cold', parts: [], role: 'assistant' }] as never)

    await opening
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('requests a fresh resume when the same main Bot Chat lost its transcript', async () => {
    $activeGatewayProfile.set('hyoseob')
    setMockAtom($selectedStoredSessionId, 'bot-chat')
    setMockAtom($activeSessionId, 'runtime-stale')
    setMockAtom($messages, [])

    const opening = host.openSession('bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000
    })

    await Promise.resolve()
    expect(requestSessionResume).toHaveBeenCalledWith('bot-chat')

    setMockAtom($activeSessionId, 'runtime-refreshed')
    setMockAtom($messages, [{ id: 'history-restored', parts: [], role: 'assistant' }] as never)

    await opening
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('resolves a history-bearing wake on transcript paint without waiting for the runtime (paint-first)', async () => {
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async (target: null | string | undefined) => {
      $activeGatewayProfile.set(target || 'default')
    })

    setMockAtom($selectedStoredSessionId, null)
    setMockAtom($activeSessionId, null)
    setMockAtom($messages, [])

    let resolved = false

    const opening = host
      .openSession('slow-runtime-chat', {
        profile: 'hyoseob',
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1_000
      })
      .then(() => {
        resolved = true
      })

    await Promise.resolve()
    await Promise.resolve()
    expect(resolved).toBe(false)

    // The REST prefetch paints the persisted transcript while the runtime
    // resume (agent build, MCP discovery, skill load) is still warming — the
    // exact cold-boot shape where the runtime takes 30s+ on slow machines.
    // The wake must complete on the paint; the runtime binds in background.
    setMockAtom($selectedStoredSessionId, 'slow-runtime-chat')
    setMockAtom($messages, [{ id: 'painted-history', parts: [], role: 'assistant' }] as never)

    await opening
    expect(resolved).toBe(true)
    expect($activeSessionId.get()).toBeNull()
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('accepts an explicitly empty Bot Chat once its main runtime is bound', async () => {
    $activeGatewayProfile.set('hyoseob')

    const opening = host.openSession('empty-bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: false,
      hydrationTimeoutMs: 1_000
    })

    setMockAtom($selectedStoredSessionId, 'empty-bot-chat')
    setMockAtom($activeSessionId, 'runtime-empty')
    setMockAtom($messages, [])

    await opening
    expect($activeSessionId.get()).toBe('runtime-empty')
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('arms the core Retry surface when Bot Chat hydration times out', async () => {
    $activeGatewayProfile.set('hyoseob')

    await expect(
      host.openSession('stranded-bot-chat', {
        profile: 'hyoseob',
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1
      })
    ).rejects.toThrow(/timed out loading/i)

    expect(setResumeExhaustedSessionId).toHaveBeenCalledWith('stranded-bot-chat')
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('lets the latest rapid bot selection win and cancels the older hydration wait', async () => {
    vi.mocked(ensureGatewayProfile).mockImplementation(async (target: null | string | undefined) => {
      $activeGatewayProfile.set(target || 'default')
    })

    const firstOutcome = host
      .openSession('chat-a', {
        profile: 'jimin',
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1_000
      })
      .then(
        () => 'resolved',
        error => String(error)
      )

    await Promise.resolve()

    const second = host.openSession('chat-b', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000
    })

    setMockAtom($selectedStoredSessionId, 'chat-b')
    setMockAtom($activeSessionId, 'runtime-b')
    setMockAtom($messages, [{ id: 'history-b', parts: [], role: 'assistant' }] as never)

    await second
    expect(await firstOutcome).toMatch(/superseded/i)
    expect($activeGatewayProfile.get()).toBe('hyoseob')
    expect($selectedStoredSessionId.get()).toBe('chat-b')
    expect($gatewaySwapTarget.get()).toBeNull()
  })
})
