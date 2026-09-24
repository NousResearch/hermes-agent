import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('./client', () => ({
  capabilityScoped: vi.fn(),
  getApiRequestConnection: vi.fn(() => 'prometheus'),
  getApiRequestProfile: vi.fn(() => null),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const {
  deleteSession,
  getSession,
  setSessionArchived,
  setSessionPinnedRemote,
  setSessionUnreadRemote,
  listSidebarSessions
} = await import('./sessions')

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
    hermesApi.mockResolvedValue({
      cron: { sessions: [] },
      errors: [{ error: 'database disk image is malformed', profile: 'default' }],
      messaging: { sessions: [] },
      recents: { sessions: [] },
      storage: { default: 'corrupt' }
    } as never)

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
