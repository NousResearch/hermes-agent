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
vi.mock('@/hermes', () => ({ deleteProfile: vi.fn(), getLogs: vi.fn(), getStatus: vi.fn(), hermesApi: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/system-actions', () => ({ runGatewayRestart: vi.fn() }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  type LineageRow = { _lineage_root_id?: null | string; id: string }

  return {
    $activeSessionId: atom(null),
    $connection: atom(null),
    $cronSessions: atom([]),
    $currentCwd: atom(''),
    $currentModel: atom(''),
    $gatewayState: atom('open'),
    $messages: atom([]),
    $messagingSessions: atom([]),
    $selectedStoredSessionId: atom(null),
    $sessions: atom([]),
    $unreadFinishedSessionIds: atom([]),
    lineageAliases: (storedId: string) => [storedId],
    rememberedSessionProfile: (_sessions: unknown, _sessionId: null | string, activeProfile: null | string) =>
      (activeProfile ?? '').trim() || 'default',
    requestSessionResume: vi.fn(),
    sessionMatchesStoredId: (session: LineageRow, storedSessionId: string) =>
      session.id === storedSessionId || session._lineage_root_id === storedSessionId,
    sessionPinId: (session: LineageRow) => session._lineage_root_id ?? session.id,
    setSessionOwnerHint: vi.fn(),
    setResumeExhaustedSessionId: vi.fn()
  }
})
vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return {
    $attentionSessionIds: atom([]),
    $draftSessionIds: atom([]),
    $focusedRuntimeId: atom(null),
    $focusedSessionState: atom(null),
    $focusedStoredSessionId: atom(null),
    $sessionTiles: atom([]),
    $sessionStates: atom({}),
    $stalledSessionIds: atom([]),
    $workingSessionIds: atom([]),
    dropTilesForProfile: vi.fn(),
    sessionTileDelegate: vi.fn(() => null)
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
    $hydrationSyncProfile: atom(null),
    $profiles: profiles,
    $showAllProfiles: atom(false),
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn(),
    newSessionInAgent: vi.fn(),
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
    activeGateway: vi.fn(() => null),
    activeGatewayConnectionId: vi.fn(() => 'local'),
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

const { BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS, DEFAULT_SESSION_HYDRATION_TIMEOUT_MS, host } = await import('./index')

const { openSession: openSessionCore } = await import('@/app/open-session')
const { deleteProfile, hermesApi } = await import('@/hermes')

const {
  activeGatewayConnectionId,
  openGatewayForAgent,
  openGatewayForProfile,
  requestGatewayForAgent,
  requestGatewayForProfile,
  retireLocalProfileGateways
} = await import('@/store/gateway')

const {
  $activeGatewayProfile,
  $gatewaySwapTarget,
  $hydrationSyncProfile,
  $profiles,
  ensureGatewayProfile,
  refreshProfiles,
  setShowAllProfiles
} = await import('@/store/profile')

const {
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionStates,
  $sessionTiles,
  sessionTileDelegate
} = await import('@/store/session-states')

const { dropTilesForProfile } = await import('@/store/session-states')

const { setWorkspaceScope } = await import('@/components/pane-shell/workspace-scope')

const {
  $activeSessionId,
  $messages,
  $selectedStoredSessionId,
  requestSessionResume,
  setSessionOwnerHint,
  setResumeExhaustedSessionId
} = await import('@/store/session')

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
  vi.mocked(sessionTileDelegate).mockReturnValue(null)
  vi.mocked(activeGatewayConnectionId).mockReturnValue('local')
  $activeGatewayProfile.set('remote-worker')
  $gatewaySwapTarget.set(null)
  setMockAtom($hydrationSyncProfile, null)
  setMockAtom($focusedRuntimeId, null)
  setMockAtom($focusedStoredSessionId, null)
  setMockAtom($focusedSessionState, null)
  setMockAtom($sessionStates, {})
  setMockAtom($sessionTiles, [])
  setMockAtom($activeSessionId, null)
  setMockAtom($selectedStoredSessionId, null)
  setMockAtom($messages, [])
  $profiles.set([profile('cached-only')])
  setWorkspaceScope('sessions')
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
    // A leftover Bot Mode tile would restore on relaunch and dial the deleted
    // profile's backend, re-creating its HERMES_HOME (#94235).
    expect(dropTilesForProfile).toHaveBeenCalledWith('worker', undefined)
  })

  it('pins an ambient SSH profile delete to the active connection and target profile', async () => {
    vi.mocked(activeGatewayConnectionId).mockReturnValue('ssh-vps')

    await host.deleteProfile('worker')

    expect(retireLocalProfileGateways).not.toHaveBeenCalled()
    expect(deleteProfile).toHaveBeenCalledWith('worker', {
      connectionId: 'ssh-vps',
      profile: 'worker'
    })
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

  it('reads and hides persisted sessions through the source primary without activating the profile', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'remote-worker',
      targetProfile: 'backend-worker'
    }

    vi.mocked(hermesApi)
      .mockResolvedValueOnce({ sessions: [{ id: 'bot-chat', profile: 'backend-worker', title: 'Bot Chat' }] })
      .mockResolvedValueOnce({ ok: true, hidden: true })

    await expect(host.listPersistedSessions(route, { profile: 'backend-worker', limit: 200 })).resolves.toMatchObject({
      sessions: [{ id: 'bot-chat' }]
    })
    await expect(
      host.setPersistedSessionHidden(route, { sessionId: 'bot-chat', profile: 'backend-worker', hidden: true })
    ).resolves.toMatchObject({ ok: true, hidden: true })

    expect(hermesApi).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        connectionId: 'source-a',
        path: expect.stringContaining('/api/profiles/sessions?')
      })
    )
    expect(hermesApi).toHaveBeenNthCalledWith(2, {
      connectionId: 'source-a',
      path: '/api/sessions/bot-chat',
      method: 'PATCH',
      body: { hidden: true, profile: 'backend-worker' }
    })
    expect(vi.mocked(hermesApi).mock.calls.every(([request]) => !('profile' in request))).toBe(true)
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('forwards an explicit timeout so long-running methods outlive the generic deadline', async () => {
    // #93911: bot_relay.deliver's backend contract tolerates ~1320s (120s turn
    // lock + a 600s turn, doubled by the bounded retry). Without a way to pass
    // that bound through, every such call died at the pool's generic 30s
    // deadline and surfaced as an unclassified failure.
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'remote-worker',
      targetProfile: 'backend-worker'
    }

    await host.requestProfile(route, 'bot_relay.deliver', { message: 'hi', profile: 'backend-worker' }, 1_320_000)

    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'source-a',
      'remote-worker',
      'bot_relay.deliver',
      { message: 'hi', profile: 'backend-worker' },
      1_320_000
    )
  })

  it('survives a backend that settles at its own ceiling, and only fails past the client deadline', async () => {
    // #93911 review follow-up (adversarial): the failure mode is not "no
    // timeout" but "a timeout equal to the backend's ceiling". Model the
    // documented worst case on a virtual clock — the turn lock wait, a full
    // timed attempt, the policy-gated re-run, and then the settlement the
    // handler still has to serialize and transport — and assert that a
    // deadline set AT the ceiling loses that race while one with margin wins.
    const LOCK_WAIT_MS = 120_000
    const ATTEMPT_MS = 600_000
    const ATTEMPTS = 2
    const CEILING_MS = LOCK_WAIT_MS + ATTEMPT_MS * ATTEMPTS
    const SETTLEMENT_MS = 1_000

    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'remote-worker',
      targetProfile: 'backend-worker'
    }

    // A gateway that answers only after the full ceiling plus settlement, and
    // aborts at whatever deadline the caller handed down.
    const backendAtItsLimit = async (
      _connectionId: string,
      _profile: string,
      _method: string,
      _params: Record<string, unknown>,
      timeoutMs?: number
    ) =>
      new Promise((resolve, reject) => {
        setTimeout(() => resolve({ reply: 'delivered', reason: 'ok' }), CEILING_MS + SETTLEMENT_MS)

        if (timeoutMs !== undefined) {
          setTimeout(() => reject(new Error('request timed out')), timeoutMs)
        }
      })

    vi.useFakeTimers()

    try {
      vi.mocked(requestGatewayForAgent).mockImplementation(backendAtItsLimit as never)

      // Deadline exactly at the ceiling: the typed settlement loses the race.
      const atCeiling = host.requestProfile(route, 'bot_relay.deliver', {}, CEILING_MS)
      const atCeilingSettled = expect(atCeiling).rejects.toThrow(/timed out/)
      await vi.advanceTimersByTimeAsync(CEILING_MS + SETTLEMENT_MS)
      await atCeilingSettled

      // Same backend, deadline with settlement margin: the answer gets through.
      const withMargin = host.requestProfile(route, 'bot_relay.deliver', {}, CEILING_MS + SETTLEMENT_MS * 180)

      const withMarginSettled = expect(withMargin).resolves.toEqual({
        reason: 'ok',
        reply: 'delivered'
      })

      await vi.advanceTimersByTimeAsync(CEILING_MS + SETTLEMENT_MS)
      await withMarginSettled
    } finally {
      vi.useRealTimers()
      vi.mocked(requestGatewayForAgent).mockReset()
    }
  })

  it('leaves callers that pass no timeout on the pool default', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'remote-worker',
      targetProfile: 'backend-worker'
    }

    await host.requestProfile(route, 'profiles.list', {})

    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'remote-worker', 'profiles.list', {})
  })

  it('fails closed when a descriptor omits connection or target profile identity', async () => {
    await expect(
      host.requestProfile(
        { connectionId: '', mode: 'remote', profile: 'worker', targetProfile: 'worker' },
        'profiles.configure',
        {}
      )
    ).rejects.toThrow(/connectionId, profile, and targetProfile/)
    await expect(
      host.requestProfile(
        { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: '' },
        'profiles.configure',
        {}
      )
    ).rejects.toThrow(/connectionId, profile, and targetProfile/)

    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
  })

  it('deletes a remote profile through its captured connection without retiring local pools', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'worker',
      targetProfile: 'backend-worker'
    }

    vi.mocked(deleteProfile).mockResolvedValueOnce({ ok: true, path: '/profiles/backend-worker' })

    await host.deleteProfile(route)

    expect(deleteProfile).toHaveBeenCalledWith('backend-worker', {
      connectionId: 'source-a',
      profile: 'worker'
    })
    expect(retireLocalProfileGateways).not.toHaveBeenCalled()
    expect(dropTilesForProfile).toHaveBeenCalledWith('worker', {
      connectionId: 'source-a',
      profile: 'worker',
      targetProfile: 'backend-worker'
    })
  })

  it('rejects a remote deletion route without a connection id', async () => {
    await expect(
      host.deleteProfile({
        connectionId: '',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'backend-worker'
      })
    ).rejects.toThrow(/connectionId, profile, and targetProfile/)

    expect(deleteProfile).not.toHaveBeenCalled()
    expect(retireLocalProfileGateways).not.toHaveBeenCalled()
  })

  it('rejects an alias to backend default before the transport helper runs', async () => {
    await expect(
      host.deleteProfile({
        connectionId: 'source-a',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'default'
      })
    ).rejects.toThrow(/default profile cannot be deleted/i)

    expect(deleteProfile).not.toHaveBeenCalled()
  })

  it('retires and deletes a local alias by its backend target profile', async () => {
    const route = {
      connectionId: 'source-local',
      mode: 'local' as const,
      profile: 'worker',
      targetProfile: 'backend-worker'
    }

    vi.mocked(deleteProfile).mockResolvedValueOnce({ ok: true, path: '/profiles/backend-worker' })

    await host.deleteProfile(route)

    expect(retireLocalProfileGateways).toHaveBeenCalledWith('backend-worker')
    expect(deleteProfile).toHaveBeenCalledWith('backend-worker', {
      connectionId: 'source-local',
      profile: 'worker'
    })
    expect(dropTilesForProfile).toHaveBeenCalledWith('worker', {
      connectionId: 'source-local',
      profile: 'worker',
      targetProfile: 'backend-worker'
    })
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
  it('captures the full owner route before opening a remote session', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    await host.openSession('remote-chat', { route })

    expect(openGatewayForAgent).toHaveBeenCalledWith(
      'source-a',
      'default',
      expect.objectContaining({ spawnPriority: 'foreground' })
    )
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect($activeGatewayProfile.get()).toBe('remote-worker')
    expect(setSessionOwnerHint).toHaveBeenCalledWith('remote-chat', route)
    expect(openSessionCore).toHaveBeenCalledWith('remote-chat', expect.any(Function), 'in-place')
  })

  it('threads an exact Bot workspace owner into the core session open', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    await host.openSession('bot-chat', {
      route,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'source-a::default'
    })

    expect(openSessionCore).toHaveBeenCalledWith('bot-chat', expect.any(Function), 'in-place', {
      ownerRoute: route,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'source-a::default'
    })
  })

  it('finishes hydration on the focused Bot tile without moving the Sessions gateway', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    const opening = host.openSession('bot-chat', {
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000,
      intent: 'tab',
      route,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:source-a::default'
    })

    await Promise.resolve()
    setMockAtom($focusedStoredSessionId, 'bot-chat')
    setMockAtom($focusedRuntimeId, 'runtime-bot-chat')
    setMockAtom($focusedSessionState, {
      messages: [{ id: 'bot-history', parts: [], role: 'assistant' }],
      storedSessionId: 'bot-chat'
    } as never)

    await opening
    expect($activeGatewayProfile.get()).toBe('remote-worker')
    expect($selectedStoredSessionId.get()).toBeNull()
  })

  it('finishes a restored Bot wake from its hydrated tile before the first pointer focus', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    const opening = host.openSession('restored-bot-chat', {
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000,
      intent: 'tab',
      route,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:source-a::default'
    })

    await Promise.resolve()
    setMockAtom($sessionTiles, [
      {
        runtimeId: 'runtime-restored',
        storedSessionId: 'restored-bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:source-a::default'
      }
    ])
    setMockAtom($sessionStates, {
      'runtime-restored': {
        messages: [{ id: 'restored-history', parts: [], role: 'assistant' }],
        storedSessionId: 'restored-bot-chat'
      }
    } as never)

    await opening
    expect($focusedStoredSessionId.get()).toBeNull()
  })

  it('strands a late Bot wake when the user returns to Sessions', async () => {
    let releaseDial: (() => void) | undefined

    vi.mocked(openGatewayForAgent).mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          releaseDial = resolve
        })
    )

    const opening = host.openSession('late-bot-chat', {
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000,
      intent: 'tab',
      route: {
        connectionId: 'source-a',
        mode: 'remote',
        profile: 'writer',
        targetProfile: 'writer'
      },
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:source-a::writer'
    })

    setWorkspaceScope('sessions')
    releaseDial?.()

    await expect(opening).rejects.toThrow(/superseded/i)
    expect(openSessionCore).not.toHaveBeenCalled()
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
  })

  it('revalidates an exact route before the one allowed hydration retry', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getProfileRoutes: vi.fn(async () => [])
    }

    await expect(
      host.openSession('removed-owner-chat', {
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1,
        intent: 'tab',
        retryHydrationTimeoutOnce: true,
        route: {
          connectionId: 'removed-source',
          mode: 'remote',
          profile: 'writer',
          targetProfile: 'writer'
        },
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:removed-source::writer'
      })
    ).rejects.toThrow(/no longer available/i)

    expect(openSessionCore).toHaveBeenCalledTimes(1)
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
  })

  it('waits until the target Bot Chat runtime and history are on the focused surface', async () => {
    vi.mocked(openGatewayForProfile).mockImplementationOnce(async () => undefined)

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

    // Flush a macrotask rather than counting microtask ticks: the profile
    // activation is now a bounded race, so the number of awaits between the
    // call and the core open is an implementation detail, and `resolved`
    // staying false through a full turn of the event loop is the stronger
    // assertion anyway.
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(openSessionCore).toHaveBeenCalledWith('bot-chat', expect.any(Function), 'in-place')
    expect(resolved).toBe(false)
    expect($gatewaySwapTarget.get()).toBe('hyoseob')

    setMockAtom($focusedStoredSessionId, 'bot-chat')
    setMockAtom($focusedRuntimeId, 'runtime-tile')
    setMockAtom($focusedSessionState, {
      messages: [{ id: 'tile-history', parts: [], role: 'assistant' }],
      storedSessionId: 'bot-chat'
    } as never)

    await opening
    expect(resolved).toBe(true)
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('requests an explicit resume on a cold open where selection has not settled (#89206)', async () => {
    vi.mocked(openGatewayForProfile).mockImplementationOnce(async () => undefined)

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

    await new Promise(resolve => setTimeout(resolve, 0))

    // The old pre-open "already selected" precondition skipped this request,
    // stranding the pane blank until timeout. It must fire now.
    expect(requestSessionResume).toHaveBeenCalledWith('cold-bot-chat', undefined)

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
    expect(requestSessionResume).toHaveBeenCalledWith('bot-chat', undefined)

    setMockAtom($activeSessionId, 'runtime-refreshed')
    setMockAtom($messages, [{ id: 'history-restored', parts: [], role: 'assistant' }] as never)

    await opening
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('refreshes an already-open Bot Chat tile instead of trusting the idle snapshot (#96183)', async () => {
    const resumeTile = vi.fn(async () => 'runtime-bot-chat')

    vi.mocked(sessionTileDelegate).mockReturnValue({ resumeTile } as never)
    $activeGatewayProfile.set('hyoseob')
    setMockAtom($sessionTiles, [{ storedSessionId: 'bot-chat' }] as never)
    setMockAtom($focusedStoredSessionId, 'bot-chat')
    setMockAtom($focusedRuntimeId, 'runtime-bot-chat')
    setMockAtom($focusedSessionState, { messages: [{ id: 'stale-history', parts: [], role: 'assistant' }] } as never)

    const opening = host.openSession('bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      forceResume: true,
      hydrationTimeoutMs: 1_000,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'hyoseob'
    })

    await opening

    expect(resumeTile).toHaveBeenCalledWith('bot-chat', { refreshTranscript: true })
    expect(requestSessionResume).not.toHaveBeenCalled()
  })

  it('forces a resume on an explicit bot switch even when a cached transcript looks healthy (#93604)', async () => {
    // Bot-switch shape from the field: the previous visit left a non-empty
    // snapshot in the session-states cache, so the surface passes every
    // health check while painting STALE messages. The heuristic alone skips
    // the resume; the explicit-navigation caller must be able to force it.
    $activeGatewayProfile.set('hyoseob')
    setMockAtom($selectedStoredSessionId, 'bot-chat')
    setMockAtom($activeSessionId, 'runtime-stale-snapshot')
    setMockAtom($messages, [{ id: 'stale-history', parts: [], role: 'assistant' }] as never)

    const opening = host.openSession('bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      forceResume: true,
      hydrationTimeoutMs: 1_000
    })

    await Promise.resolve()
    expect(requestSessionResume).toHaveBeenCalledWith('bot-chat', undefined)

    await opening
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('still trusts a healthy surface when the caller does not force a resume', async () => {
    $activeGatewayProfile.set('hyoseob')
    setMockAtom($selectedStoredSessionId, 'bot-chat')
    setMockAtom($activeSessionId, 'runtime-live')
    setMockAtom($messages, [{ id: 'live-history', parts: [], role: 'assistant' }] as never)

    const opening = host.openSession('bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000
    })

    await Promise.resolve()
    expect(requestSessionResume).not.toHaveBeenCalled()

    await opening
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('resolves a history-bearing wake on transcript paint without waiting for the runtime (paint-first)', async () => {
    vi.mocked(openGatewayForProfile).mockImplementationOnce(async () => undefined)

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

  it('keeps the ordinary session hydration default at 20 seconds', async () => {
    vi.useFakeTimers()

    try {
      expect(DEFAULT_SESSION_HYDRATION_TIMEOUT_MS).toBe(20_000)
      $activeGatewayProfile.set('hyoseob')

      const outcome = host
        .openSession('slow-bot-chat', {
          profile: 'hyoseob',
          awaitHydration: true,
          expectHistory: true
        })
        .then(
          () => 'resolved',
          error => String(error)
        )

      await vi.advanceTimersByTimeAsync(19_999)
      expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(1)
      expect(await outcome).toMatch(/timed out loading/i)
      expect(setResumeExhaustedSessionId).toHaveBeenCalledWith('slow-bot-chat')
    } finally {
      vi.useRealTimers()
    }
  })

  it('honors the exported 60-second Bot Chat hydration override', async () => {
    vi.useFakeTimers()

    try {
      expect(BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS).toBe(60_000)
      $activeGatewayProfile.set('hyoseob')

      const outcome = host
        .openSession('slow-bot-chat', {
          profile: 'hyoseob',
          awaitHydration: true,
          expectHistory: true,
          hydrationTimeoutMs: BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS
        })
        .then(
          () => 'resolved',
          error => String(error)
        )

      await vi.advanceTimersByTimeAsync(BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS - 1)
      expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(1)
      expect(await outcome).toMatch(/timed out loading/i)
      expect(setResumeExhaustedSessionId).toHaveBeenCalledWith('slow-bot-chat')
    } finally {
      vi.useRealTimers()
    }
  })

  it('retries a Bot Chat hydration timeout once without ever arming the Retry surface (#89617)', async () => {
    $activeGatewayProfile.set('hyoseob')

    // First attempt: leave the surface unhealthy so the hydration wait below
    // times out, exactly like a profile backend still waking up. Second
    // attempt: the backend is warm now — hydrate immediately. Queued as two
    // one-shots (not a standing mockImplementation) so this doesn't leak into
    // later tests via the shared afterEach's vi.clearAllMocks(), which clears
    // call history but not a standing implementation.
    vi.mocked(openSessionCore)
      .mockImplementationOnce(() => undefined)
      .mockImplementationOnce(() => {
        setMockAtom($selectedStoredSessionId, 'waking-bot-chat')
        setMockAtom($activeSessionId, 'runtime-waking')
        setMockAtom($messages, [{ id: 'history-waking', parts: [], role: 'assistant' }] as never)
      })

    await host.openSession('waking-bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1,
      retryHydrationTimeoutOnce: true
    })

    expect(openSessionCore).toHaveBeenCalledTimes(2)
    // The overlay in apps/desktop/src/app/chat/index.tsx is gated purely on
    // this atom equalling the routed session id — if it were ever set here,
    // the user would land on "Couldn't load this session" even though the
    // retry above succeeded underneath, since only a manual resumeSession()
    // call clears it for the currently-routed session.
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('still arms the Retry surface when a retried Bot Chat hydration times out twice', async () => {
    $activeGatewayProfile.set('hyoseob')

    await expect(
      host.openSession('stranded-bot-chat', {
        profile: 'hyoseob',
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1,
        retryHydrationTimeoutOnce: true
      })
    ).rejects.toThrow(/timed out loading/i)

    expect(openSessionCore).toHaveBeenCalledTimes(2)
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
    expect($activeGatewayProfile.get()).toBe('remote-worker')
    expect($selectedStoredSessionId.get()).toBe('chat-b')
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('keeps chrome API home on the previous profile when opening a Bot Chat', async () => {
    $activeGatewayProfile.set('default')

    await host.openSession('bot-chat', {
      profile: 'worker',
      keepAllProfilesScope: true
    })

    expect(ensureGatewayProfile).not.toHaveBeenCalled()
    expect(openGatewayForProfile).toHaveBeenCalledWith(
      'worker',
      expect.objectContaining({ spawnPriority: 'foreground' })
    )
    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('defaults keepAllProfilesScope to navigation instead of a workspace switch', async () => {
    $activeGatewayProfile.set('default')

    await host.openSession('bot-chat', { profile: 'worker' })

    expect(ensureGatewayProfile).not.toHaveBeenCalled()
    expect(openGatewayForProfile).toHaveBeenCalledWith(
      'worker',
      expect.objectContaining({ spawnPriority: 'foreground' })
    )
    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('still switches workspace when keepAllProfilesScope is false', async () => {
    $activeGatewayProfile.set('default')
    vi.mocked(openGatewayForProfile).mockImplementationOnce(async () => undefined)

    await host.openSession('stored-worker', {
      profile: 'worker',
      keepAllProfilesScope: false
    })

    expect(ensureGatewayProfile).toHaveBeenCalledWith('worker')
    expect(openGatewayForProfile).not.toHaveBeenCalled()
    expect(setShowAllProfiles).toHaveBeenCalledWith(false)
    expect($activeGatewayProfile.get()).toBe('worker')
  })

  it('keeps the Sessions sidebar in all-profiles even when the bot is already live', async () => {
    $activeGatewayProfile.set('hyoseob')
    setMockAtom($selectedStoredSessionId, 'bot-chat')
    setMockAtom($activeSessionId, 'runtime-live')
    setMockAtom($messages, [{ id: 'history', parts: [], role: 'assistant' }] as never)

    await host.openSession('bot-chat', {
      profile: 'hyoseob',
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000
    })

    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
  })

  it('surfaces a wedged profile activation instead of waiting on it forever (#89556)', async () => {
    // The dial the store is waiting on never settles - a backend that accepts
    // the socket and then never completes the handshake. Before this was
    // bounded, openSession's await sat here for the life of the window and the
    // hydration timer, which is armed downstream, never got a chance to fire:
    // the pane wedged with no error and no Retry rather than timing out.
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(() => new Promise<void>(() => undefined))

    setMockAtom($selectedStoredSessionId, 'wedged-chat')
    setMockAtom($activeSessionId, 'runtime-wedged')
    setMockAtom($messages, [{ id: 'history-1', parts: [], role: 'user' }] as never)

    await expect(
      host.openSession('wedged-chat', {
        profile: 'medicina',
        intent: 'main',
        awaitHydration: true,
        expectHistory: true,
        keepAllProfilesScope: false,
        hydrationTimeoutMs: 60
      })
    ).rejects.toThrow("Timed out loading medicina's session history.")

    // The stranded-session surface is the point: a bounded failure the user can
    // retry, not a frozen pane.
    expect(setResumeExhaustedSessionId).toHaveBeenCalledWith('wedged-chat')
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('names the phase in the wake log so a stuck dial is not read as a slow transcript', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    vi.mocked(ensureGatewayProfile).mockImplementationOnce(() => new Promise<void>(() => undefined))

    await expect(
      host.openSession('wedged-chat', {
        profile: 'medicina',
        intent: 'main',
        awaitHydration: true,
        expectHistory: true,
        keepAllProfilesScope: false,
        hydrationTimeoutMs: 60
      })
    ).rejects.toThrow('Timed out loading ')

    expect(warn).toHaveBeenCalledWith(
      '[bot-wake] hydration timed out',
      expect.objectContaining({ phase: 'activation' })
    )

    const payload = warn.mock.calls.at(-1)?.[1] as { hydrationWaitMs: number; profileActivationMs: number }

    // The whole budget went to activation. Reporting it as hydration wait time
    // would send a support bundle reader looking at transcript size.
    expect(payload.profileActivationMs).toBeGreaterThan(0)
    expect(payload.hydrationWaitMs).toBe(0)

    warn.mockRestore()
  })

  it('gives hydration its own full budget after a slow but successful activation', async () => {
    // Guards the choice of a SEPARATE budget per phase over one shared clock.
    // A cold profile backend can legitimately spend most of the budget getting
    // its socket up; if hydration then inherited what was left, this wake would
    // fail even though the transcript painted well inside the documented
    // 20s-equivalent window.
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(
      async (target: null | string | undefined) =>
        new Promise<void>(resolve => {
          setTimeout(() => {
            $activeGatewayProfile.set(target || 'default')
            resolve()
          }, 150)
        })
    )

    const opening = host.openSession('slow-dial-chat', {
      profile: 'medicina',
      intent: 'main',
      awaitHydration: true,
      expectHistory: true,
      keepAllProfilesScope: false,
      hydrationTimeoutMs: 200
    })

    // 300ms total is past a single shared 200ms budget and inside hydration's
    // own one, which only starts once the profile is actually active.
    await new Promise(resolve => setTimeout(resolve, 300))
    setMockAtom($selectedStoredSessionId, 'slow-dial-chat')
    setMockAtom($activeSessionId, 'runtime-medicina')
    setMockAtom($messages, [{ id: 'history-1', parts: [], role: 'user' }] as never)

    await expect(opening).resolves.toBeUndefined()
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
  })

  it('absorbs an activation that rejects after its budget has already expired', async () => {
    // The abandoned race loser keeps running - there is no cancellation handle
    // for an in-flight dial - so a late rejection must not escape as an
    // unhandled promise rejection long after the caller gave up.
    // Node's process-level hook, not window's: under jsdom a rejection that
    // escapes lands on the runner's process, which is where vitest turns it
    // into an "Unhandled Rejection" run failure.
    const unhandled: unknown[] = []

    const onUnhandled = (reason: unknown) => {
      unhandled.push(reason)
    }

    const existing = process.listeners('unhandledRejection')

    for (const listener of existing) {
      process.off('unhandledRejection', listener)
    }

    process.on('unhandledRejection', onUnhandled)

    vi.mocked(ensureGatewayProfile).mockImplementationOnce(
      () =>
        new Promise<void>((_resolve, reject) => {
          setTimeout(() => reject(new Error('dial failed after the caller gave up')), 120)
        })
    )

    await expect(
      host.openSession('late-failure-chat', {
        profile: 'medicina',
        intent: 'main',
        awaitHydration: true,
        expectHistory: true,
        keepAllProfilesScope: false,
        hydrationTimeoutMs: 40
      })
    ).rejects.toThrow('Timed out loading ')

    await new Promise(resolve => setTimeout(resolve, 200))
    process.off('unhandledRejection', onUnhandled)

    for (const listener of existing) {
      process.on('unhandledRejection', listener)
    }

    expect(unhandled).toEqual([])
  })

  it('leaves a plain open unbounded, because it has no budget and no Retry surface', async () => {
    // Deliberate scope line, not an oversight: the activation deadline rides on
    // the same contract as the hydration one. A caller that never passed
    // awaitHydration gets exactly the behaviour it had before, since a
    // rejection here would surface to code with nowhere to render it.
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(() => new Promise<void>(() => undefined))

    let settled = 'pending'

    void host
      .openSession('plain-chat', {
        profile: 'medicina',
        intent: 'main',
        keepAllProfilesScope: false,
        hydrationTimeoutMs: 40
      })
      .then(
        () => {
          settled = 'resolved'
        },
        () => {
          settled = 'rejected'
        }
      )

    await new Promise(resolve => setTimeout(resolve, 200))

    expect(settled).toBe('pending')
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
  })
})

describe('shared-remote hydration gate (#89843)', () => {
  it('paints stored history immediately instead of holding the wake on an unsatisfiable profile gate', async () => {
    $activeGatewayProfile.set('default')
    // Shared-remote: every profile is served through the primary socket, so
    // the dial resolves but $activeGatewayProfile NEVER moves to the bot's
    // profile — the old profileMatches gate could not be satisfied and the
    // wake burned the whole 20s budget with the transcript already painted.
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async () => undefined)

    const opening = host.openSession('shared-remote-bot', {
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 250,
      keepAllProfilesScope: false,
      profile: 'shadow'
    })

    await Promise.resolve()
    setMockAtom($selectedStoredSessionId, 'shared-remote-bot')
    setMockAtom($activeSessionId, 'runtime-shared-remote')
    setMockAtom($messages, [{ id: 'stored-history', parts: [], role: 'assistant' }] as never)

    // Must resolve on the painted transcript, well inside the budget.
    await opening
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()

    // The wake resolved paint-first, so the subtle syncing affordance is up…
    expect($hydrationSyncProfile.get()).toBe('shadow')

    // …and clears itself when the profile gate finally catches up.
    $activeGatewayProfile.set('shadow')
    expect($hydrationSyncProfile.get()).toBeNull()
  })

  it('still fails closed for an expected-empty chat when the profile gate stays unsatisfied', async () => {
    $activeGatewayProfile.set('default')
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async () => undefined)

    // No transcript to paint: a bound runtime on the WRONG profile is not
    // proof of a real surface, so the paint-first bypass must not fire.
    setMockAtom($selectedStoredSessionId, 'shared-remote-empty')
    setMockAtom($activeSessionId, 'runtime-empty')

    await expect(
      host.openSession('shared-remote-empty', {
        awaitHydration: true,
        expectHistory: false,
        hydrationTimeoutMs: 40,
        keepAllProfilesScope: false,
        profile: 'shadow'
      })
    ).rejects.toThrow(/timed out loading/i)

    expect($hydrationSyncProfile.get()).toBeNull()
  })

  it('never paint-first resolves a superseded wake (fail closed on conflicting concurrent hydration)', async () => {
    $activeGatewayProfile.set('default')
    vi.mocked(ensureGatewayProfile).mockImplementation(async () => undefined)

    const firstOutcome = host
      .openSession('conflict-a', {
        awaitHydration: true,
        expectHistory: true,
        hydrationTimeoutMs: 1_000,
        keepAllProfilesScope: false,
        profile: 'shadow'
      })
      .then(
        () => 'resolved',
        error => String(error)
      )

    await Promise.resolve()

    const second = host.openSession('conflict-b', {
      awaitHydration: true,
      expectHistory: true,
      hydrationTimeoutMs: 1_000,
      keepAllProfilesScope: false,
      profile: 'zephyr'
    })

    setMockAtom($selectedStoredSessionId, 'conflict-b')
    setMockAtom($activeSessionId, 'runtime-conflict-b')
    setMockAtom($messages, [{ id: 'history-conflict-b', parts: [], role: 'assistant' }] as never)

    await second
    expect(await firstOutcome).toMatch(/superseded/i)
    // Only the CURRENT wake's profile is syncing — the superseded one never
    // painted its badge over the winner.
    expect($hydrationSyncProfile.get()).toBe('zephyr')
  })
})
