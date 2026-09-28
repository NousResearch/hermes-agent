import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionMessagesResponse } from '@/types/hermes'

import type { SidebarSessionsResponse } from './sessions'

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  // The cross-backend probe reads the session's own route too: `getLatestSessionMessages`
  // spreads both scope selectors before it dials.
  ambientOwnerConnectionId: vi.fn(() => 'local'),
  connectionScoped: vi.fn(() => ({})),
  getApiRequestConnection: vi.fn(() => 'prometheus'),
  getApiRequestProfile: vi.fn(() => null),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const {
  deleteSession,
  fetchStoredTranscriptAcrossBackends,
  getSession,
  setSessionArchived,
  setSessionPinnedRemote,
  setSessionUnreadRemote,
  listSidebarSessions
} = await import('./sessions')

const { $connectionsRegistry } = await import('@/store/connection-registry-state')

const hermesApi = vi.mocked(client.hermesApi)

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(client.getApiRequestConnection).mockReturnValue('prometheus')
  vi.mocked(client.getApiRequestProfile).mockReturnValue(null)
})

describe('deleteSession profile scoping', () => {
  it('scopes the DELETE to the owning profile in the URL (object owner)', async () => {
    // Regression: the sidebar "All Profiles" delete sent the profile only via
    // request.profile, not in the URL. On a remote gateway with no remoteProfile
    // alias the main-process path rewrite left the URL unscoped, so the backend
    // opened its own default state.db, missed the row, and returned
    // {ok:true, already_absent:true} — the row vanished optimistically but was
    // never deleted and came back on refresh. The URL must carry ?profile=.
    hermesApi.mockResolvedValue({ ok: true } as never)
    // Mirrors the real capabilityScoped for an object owner (remote-stamped row).
    vi.mocked(client.capabilityScoped).mockReturnValue({ profile: 'tommy', connectionId: 'hermes-pi' })

    await deleteSession('sess-1', { connectionId: 'hermes-pi', profile: 'tommy' })

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'DELETE',
      path: '/api/sessions/sess-1?profile=tommy',
      connectionId: 'hermes-pi',
      profile: 'tommy'
    })
  })

  it('scopes the DELETE to the owning profile in the URL (bare string owner)', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)
    // Bare-string owner: capabilityScoped resolves it to a profile scope.
    vi.mocked(client.capabilityScoped).mockReturnValue({ profile: 'tommy' })

    await deleteSession('sess-2', 'tommy')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'DELETE',
      path: '/api/sessions/sess-2?profile=tommy'
    })
  })

  it('omits the profile query when no owner is known', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await deleteSession('sess-3')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'DELETE',
      path: '/api/sessions/sess-3'
    })
    expect((hermesApi.mock.calls[0][0] as { path: string }).path).not.toContain('profile=')
  })

  it('keeps an explicit local pin routed to the local pool', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)
    // capabilityScoped drops a 'local' connection id by design; sessionScoped
    // must re-add it so the request stays pinned to this device.
    vi.mocked(client.capabilityScoped).mockReturnValue({ profile: 'tommy' })

    await deleteSession('sess-4', { connectionId: 'local', profile: 'tommy' })

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'DELETE',
      path: '/api/sessions/sess-4?profile=tommy',
      connectionId: 'local',
      profile: 'tommy'
    })
  })
})

describe('getSession dial priority', () => {
  it('does not dial an explicitly scoped session read foreground', async () => {
    // The scope helper tags every explicit scope foreground (#111651); the
    // cross-profile probe loop in resolveStoredSession would otherwise cold-start
    // every other profile on the reserved slot during a boot-time resume.
    hermesApi.mockResolvedValue({ id: 'sess-5' } as never)
    vi.mocked(client.capabilityScoped).mockReturnValue({ priority: 'foreground', profile: 'tommy' })

    await getSession('sess-5', { connectionId: 'local', profile: 'tommy' })

    expect(hermesApi.mock.calls[0][0]).toMatchObject({ profile: 'tommy', connectionId: 'local' })
    expect(hermesApi.mock.calls[0][0]).not.toHaveProperty('priority')
  })
})

describe('setSessionArchived profile scoping', () => {
  it('carries the owning profile in the PATCH body', async () => {
    // Same class as the unscoped DELETE: the PATCH handler reads its target DB
    // from body.profile, so archiving a foreign-profile session must send it in
    // the body, not only as request.profile (Electron routing), or on a remote
    // gateway the archive lands on the wrong state.db and silently no-ops.
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionArchived('sess-a', true, 'tommy')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'PATCH',
      path: '/api/sessions/sess-a',
      profile: 'tommy',
      body: { archived: true, profile: 'tommy' }
    })
  })

  it('falls back to the ACTIVE profile in the body when no owner is given', async () => {
    // Multiplex-only: the PATCH handler resolves its state.db from
    // `body.profile` and there is no per-profile backend whose HERMES_HOME
    // could stand in. An unnamed owner therefore has to mean "the profile I am
    // looking at" — otherwise the archive lands on the shared backend's own
    // state.db and silently no-ops.
    hermesApi.mockResolvedValue({ ok: true } as never)
    vi.mocked(client.getApiRequestProfile).mockReturnValue('beta')

    await setSessionArchived('sess-b', false)

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'PATCH',
      profile: 'beta',
      body: { archived: false, profile: 'beta' }
    })
  })

  it('omits the profile from the body only when there is no active profile at all', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionArchived('sess-b2', false)

    const req = hermesApi.mock.calls[0][0] as { body: Record<string, unknown> }
    expect(req).toMatchObject({ method: 'PATCH', body: { archived: false } })
    expect(req.body).not.toHaveProperty('profile')
  })
})

describe('setSessionPinnedRemote / setSessionUnreadRemote profile scoping', () => {
  it('carries the owning profile in the pin PATCH body', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionPinnedRemote('sess-p', true, 'tommy')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'PATCH',
      path: '/api/sessions/sess-p',
      profile: 'tommy',
      body: { pinned: true, profile: 'tommy' }
    })
  })

  it('carries the owning profile in the unread PATCH body', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionUnreadRemote('sess-u', true, 'tommy')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'PATCH',
      path: '/api/sessions/sess-u',
      profile: 'tommy',
      body: { unread: true, profile: 'tommy' }
    })
  })

  it('falls back to the ACTIVE profile in the body when no owner is given', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)
    vi.mocked(client.getApiRequestProfile).mockReturnValue('beta')

    await setSessionPinnedRemote('sess-p2', false)

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      method: 'PATCH',
      profile: 'beta',
      body: { pinned: false, profile: 'beta' }
    })
  })
})

describe('listSidebarSessions remote ownership', () => {
  it('stamps active remote rows so a later resume stays on their gateway', async () => {
    hermesApi.mockResolvedValue({
      cron: { sessions: [] },
      messaging: { sessions: [] },
      recents: {
        sessions: [{ id: 'remote-session', profile: 'default', source: 'desktop', title: 'Remote chat' }]
      }
    } as never)

    const result = await listSidebarSessions({
      recentsProfile: 'default',
      recentsLimit: 40,
      recentsExclude: [],
      cronLimit: 20,
      messagingLimit: 40,
      messagingExclude: []
    })

    expect(result.recents.sessions[0]).toMatchObject({ connection_id: 'prometheus', id: 'remote-session' })
  })
})

describe('listSidebarSessions storage health', () => {
  it('passes the backend corrupt-store map through so the sidebar can say why it is empty', async () => {
    const response = {
      cron: { sessions: [] },
      errors: [{ error: 'database disk image is malformed', profile: 'default' }],
      messaging: { sessions: [] },
      recents: { sessions: [] },
      storage: { default: 'corrupt' }
    } satisfies SidebarSessionsResponse

    // SAFETY: vi cannot infer a concrete return from the generic hermesApi signature;
    // `satisfies` above checks the exact endpoint contract before it crosses the mock boundary.
    hermesApi.mockResolvedValue(response as never)

    const result = await listSidebarSessions({
      recentsProfile: 'all',
      recentsLimit: 40,
      recentsExclude: [],
      cronLimit: 20,
      messagingLimit: 40,
      messagingExclude: []
    })

    expect(result.storage).toEqual({ default: 'corrupt' })
  })
})

/**
 * #94724 no-owner recovery: the read-only cross-backend probe.
 *
 * The PR that normalized this 404 into a typed error was closed: the signal the
 * routing needs — "a miss on the ambient store is not proof of absence, keep
 * looking" — lives in the control flow, not in the exception. These tests pin
 * that control flow, including the symmetric control the review asked for
 * (every backend missing lands on the documented aggregate `null`).
 */
describe('fetchStoredTranscriptAcrossBackends (#94724 no-owner recovery)', () => {
  const page = (sessionId: string, text: string) =>
    ({ messages: [{ content: text, role: 'user' }], session_id: sessionId }) as unknown as SessionMessagesResponse

  const notFound = () => new Error('404: {"detail":"Session not found"}')

  beforeEach(() => {
    vi.mocked(client.capabilityScoped).mockImplementation(scope =>
      scope && typeof scope === 'object'
        ? {
            ...(scope.profile?.trim() ? { profile: scope.profile.trim() } : {}),
            ...(scope.connectionId?.trim() ? { connectionId: scope.connectionId.trim() } : {}),
            priority: 'foreground'
          }
        : { priority: 'foreground' }
    )
  })

  afterEach(() => {
    $connectionsRegistry.set(null)
  })

  it('serves the ambient transcript without probing any registered backend', async () => {
    $connectionsRegistry.set({ connections: [{ id: 'backend-b' }] } as never)
    hermesApi.mockResolvedValueOnce(page('sess-1', 'ambient') as never)

    const result = await fetchStoredTranscriptAcrossBackends('sess-1')

    expect(result).toMatchObject({ session_id: 'sess-1' })
    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      path: expect.stringContaining('/api/sessions/sess-1/messages')
    })
  })

  it('keeps probing a registered backend when the ambient read 404s', async () => {
    // The routing signal must survive the first probe: a miss there is a plain
    // 404 (no session minted, no live route), so the search continues by id.
    $connectionsRegistry.set({ connections: [{ id: 'backend-b' }] } as never)
    hermesApi.mockImplementation(
      ((request: { connectionId?: string }) =>
        request?.connectionId === 'backend-b'
          ? Promise.resolve(page('sess-1', 'from-b'))
          : Promise.reject(notFound())) as never
    )

    const result = await fetchStoredTranscriptAcrossBackends('sess-1')

    expect(result).toMatchObject({ session_id: 'sess-1' })
    expect(hermesApi).toHaveBeenCalledTimes(2)
    expect(hermesApi.mock.calls[1][0]).toMatchObject({
      connectionId: 'backend-b',
      path: expect.stringContaining('/api/sessions/sess-1/messages')
    })
  })

  it('probes past an unreachable backend and still finds the transcript', async () => {
    $connectionsRegistry.set({ connections: [{ id: 'wedged' }, { id: 'backend-b' }] } as never)
    hermesApi.mockImplementation(
      ((request: { connectionId?: string }) =>
        request?.connectionId === 'backend-b'
          ? Promise.resolve(page('sess-1', 'from-b'))
          : Promise.reject(new Error('connect ECONNREFUSED 127.0.0.1:6262'))) as never
    )

    const result = await fetchStoredTranscriptAcrossBackends('sess-1')

    expect(result).toMatchObject({ session_id: 'sess-1' })
    expect(hermesApi).toHaveBeenCalledTimes(3)
  })

  it('skips the ambient id and local, and returns the aggregate null when every probe misses', async () => {
    $connectionsRegistry.set({ connections: [{ id: 'prometheus' }, { id: 'local' }, { id: 'backend-b' }] } as never)
    hermesApi.mockRejectedValue(notFound())

    const result = await fetchStoredTranscriptAcrossBackends('sess-1')

    // 'prometheus' is the ambient connection (already read first) and 'local' is
    // the current window's own backend: neither is probed a second time.
    expect(result).toBeNull()
    expect(hermesApi).toHaveBeenCalledTimes(2)
    expect(hermesApi.mock.calls[1][0]).toMatchObject({ connectionId: 'backend-b' })
  })

  it('returns the documented aggregate null when every backend is unreachable, never throwing', async () => {
    $connectionsRegistry.set({ connections: [{ id: 'wedged' }] } as never)
    hermesApi.mockRejectedValue(new Error('connect ECONNREFUSED 127.0.0.1:6262'))

    await expect(fetchStoredTranscriptAcrossBackends('sess-1')).resolves.toBeNull()
  })

  it('returns null without dialing when no connection registry is installed', async () => {
    $connectionsRegistry.set(null)
    hermesApi.mockRejectedValue(notFound())

    await expect(fetchStoredTranscriptAcrossBackends('sess-1')).resolves.toBeNull()
    expect(hermesApi).toHaveBeenCalledTimes(1)
  })
})
