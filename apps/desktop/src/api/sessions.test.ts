import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { recordTranscriptTail } from '@/store/transcript-tail'

import type { SidebarSessionsResponse } from './sessions'

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('./client', () => ({
  ambientOwnerConnectionId: vi.fn(),
  capabilityScoped: vi.fn(),
  connectionScoped: vi.fn(() => ({})),
  getApiRequestConnection: vi.fn(() => 'prometheus'),
  getApiRequestProfile: vi.fn(() => null),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const {
  deleteSession,
  getSession,
  getSessionMessages,
  getLatestSessionMessages,
  setSessionArchived,
  setSessionPinnedRemote,
  setSessionUnreadRemote,
  setSessionOwnerResolver,
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

describe('unscoped session reads resolve the owner before dispatch', () => {
  // Regression (#125372): a detail/messages read that carried no caller scope
  // inherited the window's ambient connection tag, so with two connections
  // exposing the same profile name the read landed on whichever machine the
  // window was activated on and the other machine's session answered 404.
  // The store registers the owner ladder via setSessionOwnerResolver; an
  // unscoped read must consult it, an explicit scope must not be overridden.
  const identityCapabilityScoped = () =>
    vi.mocked(client.capabilityScoped).mockImplementation(
      // Mirrors the real capabilityScoped for object scopes: pass profile and
      // connectionId through (the real one also adds priority, irrelevant here).
      scope => (typeof scope === 'object' && scope !== null ? { ...scope } : scope) as never
    )

  afterEach(() => {
    setSessionOwnerResolver(undefined)
  })

  it('routes an unscoped getSession through the resolved owner route', async () => {
    hermesApi.mockResolvedValue({ id: '20260926_223922' } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => ({ connectionId: 'dale-home-lan-9119', profile: 'default' }))

    await getSession('20260926_223922')

    expect(client.capabilityScoped).toHaveBeenCalledWith({ connectionId: 'dale-home-lan-9119', profile: 'default' })
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      path: '/api/sessions/20260926_223922?profile=default',
      connectionId: 'dale-home-lan-9119',
      profile: 'default'
    })
  })

  it('prefers the route targetProfile over the desktop-side profile name', async () => {
    hermesApi.mockResolvedValue({ id: 's1' } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => ({ connectionId: 'cloud', profile: 'desktop-name', targetProfile: 'backend-name' }))

    await getSession('s1')

    expect(client.capabilityScoped).toHaveBeenCalledWith({ connectionId: 'cloud', profile: 'backend-name' })
  })

  it('keeps an explicit local pin explicit when resolving the owner', async () => {
    hermesApi.mockResolvedValue({ id: 's1' } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'default' }))

    await getSession('s1')

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      path: '/api/sessions/s1?profile=default',
      connectionId: 'local',
      profile: 'default'
    })
  })

  it('routes a string owner by profile name alone', async () => {
    hermesApi.mockResolvedValue({ id: 's2' } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => 'invest')

    await getSession('s2')

    expect(client.capabilityScoped).toHaveBeenCalledWith('invest')
  })

  it('keeps the ambient path when no owner is known', async () => {
    hermesApi.mockResolvedValue({ id: 's3' } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => undefined)

    await getSession('s3')

    // sessionScoped short-circuits an unknown owner to the ambient path —
    // no scope helper runs, no scope key rides the request.
    expect(client.capabilityScoped).not.toHaveBeenCalled()
    expect(hermesApi.mock.calls[0][0]).not.toHaveProperty('connectionId')
    expect(hermesApi.mock.calls[0][0]).not.toHaveProperty('profile')
    expect((hermesApi.mock.calls[0][0] as { path: string }).path).not.toContain('profile=')
  })

  it('never consults the resolver when the caller passed an explicit scope', async () => {
    hermesApi.mockResolvedValue({ id: 's4' } as never)
    identityCapabilityScoped()
    const resolver = vi.fn(() => ({ connectionId: 'dale-home-lan-9119', profile: 'default' }))

    setSessionOwnerResolver(resolver)

    await getSession('s4', 'other-profile')

    expect(resolver).not.toHaveBeenCalled()
    expect(client.capabilityScoped).toHaveBeenCalledWith('other-profile')
  })

  it('routes an unscoped getSessionMessages through the resolved owner route', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)
    identityCapabilityScoped()
    setSessionOwnerResolver(() => ({ connectionId: 'dale-home-lan-9119', profile: 'default' }))

    await getSessionMessages('s5', undefined, { limit: 10, order: 'latest' })

    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      connectionId: 'dale-home-lan-9119',
      profile: 'default'
    })
    expect((hermesApi.mock.calls[0][0] as { path: string }).path).toBe(
      '/api/sessions/s5/messages?profile=default&limit=10&order=latest'
    )
  })
})

describe('getLatestSessionMessages keys the tail ledger by the effective owner', () => {
  // Regression (#125372 follow-up): the READ routes an unscoped caller through
  // the owner ladder, but the tail ledger entry was keyed by the caller's
  // ambient scope. With two connections both exposing profile `default`, the
  // page came from the owner machine while the ledger recorded it under the
  // ambient one — the reader's owner-keyed lookup then missed, and an unscoped
  // resolve saw two entries for one stored id and could not pick.
  const identityCapabilityScoped = () =>
    vi.mocked(client.capabilityScoped).mockImplementation(
      // Mirrors the real capabilityScoped for object scopes (minus priority,
      // which sessionScoped strips anyway).
      scope => (typeof scope === 'object' && scope !== null ? { ...scope } : scope) as never
    )

  const recordTail = vi.mocked(recordTranscriptTail)

  beforeEach(() => {
    vi.mocked(client.connectionScoped).mockReturnValue({})
    vi.mocked(client.ambientOwnerConnectionId).mockReturnValue(undefined)
  })

  afterEach(() => {
    setSessionOwnerResolver(undefined)
  })

  it('records the entry under the owner scope the unscoped read routed by', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)
    identityCapabilityScoped()
    // Window ambient tag is MACHINE-A; the session's owner is MACHINE-B.
    vi.mocked(client.connectionScoped).mockReturnValue({ connectionId: 'machine-a' })
    vi.mocked(client.getApiRequestProfile).mockReturnValue('default')
    setSessionOwnerResolver(() => ({ connectionId: 'machine-b', profile: 'default' }))

    await getLatestSessionMessages('stored')

    // The read went to the owner machine…
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      connectionId: 'machine-b',
      profile: 'default'
    })
    // …so the ledger must key by that same machine, not the ambient one.
    expect(recordTail).toHaveBeenCalledWith(
      'stored',
      expect.anything(),
      expect.objectContaining({ connectionId: 'machine-b', profile: 'default' }),
      expect.objectContaining({ connectionId: 'machine-b', profile: 'default' })
    )
  })

  it('keeps the ambient ledger key when no owner is known', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)
    identityCapabilityScoped()
    vi.mocked(client.connectionScoped).mockReturnValue({ connectionId: 'machine-a' })
    vi.mocked(client.getApiRequestProfile).mockReturnValue('default')
    setSessionOwnerResolver(() => undefined)

    await getLatestSessionMessages('stored')

    expect(hermesApi.mock.calls[0][0]).not.toHaveProperty('connectionId')
    expect(recordTail).toHaveBeenCalledWith(
      'stored',
      expect.anything(),
      expect.objectContaining({ connectionId: 'machine-a' }),
      expect.objectContaining({ connectionId: 'machine-a', profile: 'default' })
    )
  })

  it('never consults the resolver when the caller passed an explicit scope', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)
    identityCapabilityScoped()
    vi.mocked(client.connectionScoped).mockReturnValue({ connectionId: 'machine-a' })
    const resolver = vi.fn(() => ({ connectionId: 'machine-b', profile: 'default' }))

    setSessionOwnerResolver(resolver)

    await getLatestSessionMessages('stored', { connectionId: 'machine-pinned', profile: 'work' })

    expect(resolver).not.toHaveBeenCalled()
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      connectionId: 'machine-pinned',
      profile: 'work'
    })
    expect(recordTail).toHaveBeenCalledWith(
      'stored',
      expect.anything(),
      expect.objectContaining({ connectionId: 'machine-pinned', profile: 'work' }),
      expect.objectContaining({ connectionId: 'machine-pinned', profile: 'work' })
    )
  })
})
