import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  getApiRequestConnection: vi.fn(() => 'prometheus'),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const { deleteSession, setSessionArchived, setSessionPinnedRemote, setSessionUnreadRemote, listSidebarSessions } =
  await import('./sessions')

const hermesApi = vi.mocked(client.hermesApi)

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(client.getApiRequestConnection).mockReturnValue('prometheus')
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

  it('omits the profile from the body when none is given', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionArchived('sess-b', false)

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

  it('omits the profile from the body when none is given', async () => {
    hermesApi.mockResolvedValue({ ok: true } as never)

    await setSessionPinnedRemote('sess-p2', false)

    const req = hermesApi.mock.calls[0][0] as { body: Record<string, unknown> }
    expect(req).toMatchObject({ method: 'PATCH', body: { pinned: false } })
    expect(req.body).not.toHaveProperty('profile')
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
