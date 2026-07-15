import { describe, expect, it, vi } from 'vitest'

import {
  createAppliedConnectionConfig,
  modeIsRemoteLike,
  pathWithGlobalRemoteProfile,
  profileLocalOverride,
  profileRemoteOverride
} from './connection-config'
import {
  type BackendSelection,
  connectionApplyAffectsPoolProfile,
  createProfileAsyncQueue,
  ensureBackendLifecycle,
  ensureCompatiblePoolEntry,
  mergeRemoteProfileSessionAggregates,
  planProfilePromotion,
  requestJsonForBackendConnection,
  requestJsonForProfileRoute,
  selectBackendSelection,
  shouldReuseBackendRoute
} from './profile-request-routing'

interface RecordedRequest {
  selection: BackendSelection
  url: string
}

function requestHarness(connections: Record<string, { authMode: string; baseUrl: string; token: null | string }>) {
  const selections: BackendSelection[] = []
  const requestOauth = vi.fn(async (url: string) => ({ selection: selections.at(-1), url }))
  const requestToken = vi.fn(async (url: string) => ({ selection: selections.at(-1), url }))
  const startPrimary = vi.fn(async () => connections['primary:remote-global'])
  const ensurePool = vi.fn(async (_profile: string, route: BackendSelection['route']) => connections[`pool:${route}`])

  const ensureBackend = vi.fn((profile?: null | string, options = {}) => {
    const selectedProfile = profile || 'default'

    if (options.selection) {
      selections.push(options.selection)
    }

    return ensureBackendLifecycle(selectedProfile, options, {
      ensurePool,
      // Deliberately conflict with explicit request selections. If the real
      // lifecycle seam discards one, it falls through to the remote primary.
      resolveCurrentSelection: () => ({ route: 'remote-global', target: 'primary' }),
      runExclusive: async (_queuedProfile, operation) => operation(),
      startPrimary
    })
  })

  return { ensureBackend, ensurePool, requestOauth, requestToken, startPrimary }
}

describe('profile REST request routing', () => {
  it('uses an auxiliary local request leg for the primary under an active global remote', async () => {
    const config = {
      mode: 'remote',
      profiles: { writer: { mode: 'local' } }
    }

    const dependencies = requestHarness({
      'pool:local': { authMode: 'token', baseUrl: 'http://127.0.0.1:4123', token: 'local-token' },
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }
    })

    const result = await requestJsonForProfileRoute<RecordedRequest>(
      {
        explicitLocal: profileLocalOverride(config, 'writer'),
        globalRemote: modeIsRemoteLike(config.mode),
        method: 'GET',
        path: pathWithGlobalRemoteProfile('/api/plugins/kanban/boards', 'writer', {
          globalRemote: true,
          profileLocalOverride: true
        }),
        primaryProfile: 'writer',
        profile: 'writer',
        timeoutMs: 15_000
      },
      dependencies
    )

    expect(dependencies.ensureBackend).toHaveBeenCalledWith('writer', {
      selection: { route: 'local', target: 'pool' }
    })
    expect(dependencies.ensurePool).toHaveBeenCalledWith('writer', 'local')
    expect(dependencies.startPrimary).not.toHaveBeenCalled()
    expect(dependencies.requestToken).toHaveBeenCalledWith(
      'http://127.0.0.1:4123/api/plugins/kanban/boards',
      'local-token',
      expect.objectContaining({ method: 'GET' })
    )
    expect(result.url).toBe('http://127.0.0.1:4123/api/plugins/kanban/boards')
  })

  it('uses a pooled local request leg for a non-primary explicit-local profile', async () => {
    const dependencies = requestHarness({
      'pool:local': { authMode: 'token', baseUrl: 'http://127.0.0.1:4789', token: 'pool-token' },
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }
    })

    const result = await requestJsonForProfileRoute<RecordedRequest>(
      {
        explicitLocal: true,
        globalRemote: true,
        path: '/api/sessions',
        primaryProfile: 'default',
        profile: 'writer',
        timeoutMs: 15_000
      },
      dependencies
    )

    expect(dependencies.ensureBackend).toHaveBeenCalledWith('writer', {
      selection: { route: 'local', target: 'pool' }
    })
    expect(result.url).toBe('http://127.0.0.1:4789/api/sessions')
  })

  it("treats gatewayId='local' as authoritative over a conflicting remote primary", async () => {
    const dependencies = requestHarness({
      'pool:local': { authMode: 'token', baseUrl: 'http://127.0.0.1:4333', token: 'local-token' },
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }
    })

    const result = await requestJsonForProfileRoute<RecordedRequest>(
      {
        explicitRemote: true,
        gatewayId: 'local',
        globalRemote: true,
        path: '/api/plugins/kanban/boards',
        primaryProfile: 'writer',
        profile: 'writer',
        timeoutMs: 15_000
      },
      dependencies
    )

    expect(dependencies.ensureBackend).toHaveBeenCalledWith('writer', {
      selection: { route: 'local', target: 'pool' }
    })
    expect(dependencies.ensurePool).toHaveBeenCalledWith('writer', 'local')
    expect(dependencies.startPrimary).not.toHaveBeenCalled()
    expect(dependencies.requestOauth).not.toHaveBeenCalled()
    expect(result.url).toBe('http://127.0.0.1:4333/api/plugins/kanban/boards')
  })

  it('fails closed for gateway ids without a current profile mapping', async () => {
    const dependencies = requestHarness({
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }
    })

    await expect(
      requestJsonForProfileRoute(
        {
          gatewayId: 'invented-peer',
          globalRemote: true,
          path: '/api/status',
          primaryProfile: 'default',
          timeoutMs: 15_000
        },
        dependencies
      )
    ).rejects.toThrow('Unknown gateway target: invented-peer')

    expect(dependencies.ensureBackend).not.toHaveBeenCalled()
    expect(dependencies.startPrimary).not.toHaveBeenCalled()
  })

  it('preserves cloud/global OAuth routing and its profile-scoped request path', async () => {
    const config = { mode: 'cloud' }

    const dependencies = requestHarness({
      'pool:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null },
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://primary.example', token: null }
    })

    const path = pathWithGlobalRemoteProfile('/api/model/info', 'writer', {
      globalRemote: modeIsRemoteLike(config.mode)
    })

    const result = await requestJsonForProfileRoute<RecordedRequest>(
      {
        globalRemote: modeIsRemoteLike(config.mode),
        path,
        primaryProfile: 'default',
        profile: 'writer',
        timeoutMs: 15_000
      },
      dependencies
    )

    expect(dependencies.ensureBackend).toHaveBeenCalledWith('writer', {
      selection: { route: 'remote-global', target: 'pool' }
    })
    expect(dependencies.requestOauth).toHaveBeenCalledWith(
      'https://cloud.example/api/model/info?profile=writer',
      expect.objectContaining({ timeoutMs: 15_000 })
    )
    expect(dependencies.requestToken).not.toHaveBeenCalled()
    expect(dependencies.startPrimary).not.toHaveBeenCalled()
    expect(result.url).toBe('https://cloud.example/api/model/info?profile=writer')
  })

  it('keeps the primary on its OAuth cloud lifecycle', async () => {
    const dependencies = requestHarness({
      'primary:remote-global': { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }
    })

    const result = await requestJsonForProfileRoute<RecordedRequest>(
      {
        globalRemote: true,
        path: '/api/model/info?profile=default',
        primaryProfile: 'default',
        profile: 'default',
        timeoutMs: 15_000
      },
      dependencies
    )

    expect(dependencies.ensureBackend).toHaveBeenCalledWith('default', {
      selection: { route: 'remote-global', target: 'primary' }
    })
    expect(dependencies.requestOauth).toHaveBeenCalledWith(
      'https://cloud.example/api/model/info?profile=default',
      expect.objectContaining({ timeoutMs: 15_000 })
    )
    expect(dependencies.startPrimary).toHaveBeenCalledOnce()
    expect(dependencies.ensurePool).not.toHaveBeenCalled()
    expect(result.url).toBe('https://cloud.example/api/model/info?profile=default')
  })

  it('merges an explicit remote profile through an OAuth-aware cloud primary request', async () => {
    const primary = { authMode: 'oauth', baseUrl: 'https://cloud.example', token: null }

    const requestOauth = vi.fn(async () => ({
      profile_totals: { default: 1, writer: 1 },
      sessions: [
        { id: 'primary', last_active: 20, profile: 'default' },
        { id: 'stale-writer', last_active: 10, profile: 'writer' }
      ],
      total: 2
    }))

    const requestToken = vi.fn()

    const requestRemoteProfile = vi.fn(async () => ({
      sessions: [{ id: 'remote-writer', last_active: 30, profile: 'writer' }],
      total: 1
    }))

    const result = await mergeRemoteProfileSessionAggregates(
      new URLSearchParams('profile=all&limit=20&offset=0'),
      ['writer'],
      {
        requestPrimary: path =>
          requestJsonForBackendConnection(
            primary,
            path,
            { method: 'GET', timeoutMs: 15_000 },
            {
              requestOauth,
              requestToken
            }
          ),
        requestRemoteProfile
      }
    )

    expect(requestOauth).toHaveBeenCalledWith(
      'https://cloud.example/api/profiles/sessions?profile=all&limit=20&offset=0',
      expect.objectContaining({ method: 'GET' })
    )
    expect(requestToken).not.toHaveBeenCalled()
    expect(requestRemoteProfile).toHaveBeenCalledWith('writer', new URLSearchParams('profile=all&limit=20&offset=0'))
    expect(result.sessions.map(session => session.id)).toEqual(['remote-writer', 'primary'])
    expect(result.profile_totals).toEqual({ default: 1, writer: 1 })
    expect(result.total).toBe(2)
  })

  it('propagates primary OAuth auth failures instead of returning an empty aggregate', async () => {
    const requestRemoteProfile = vi.fn()

    await expect(
      mergeRemoteProfileSessionAggregates(new URLSearchParams('profile=all'), ['writer'], {
        requestPrimary: async () => {
          throw new Error('401 OAuth session expired')
        },
        requestRemoteProfile
      })
    ).rejects.toThrow('401 OAuth session expired')

    expect(requestRemoteProfile).not.toHaveBeenCalled()
  })

  it('keeps ordinary local primary traffic on the primary lifecycle', () => {
    expect(
      selectBackendSelection({
        globalRemote: false,
        primaryProfile: 'default',
        profile: 'default'
      })
    ).toEqual({ route: 'local', target: 'primary' })

    expect(
      selectBackendSelection({
        forceLocal: true,
        globalRemote: false,
        primaryProfile: 'default',
        profile: 'default'
      })
    ).toEqual({ route: 'local', target: 'pool' })
  })

  it('invalidates pool entries affected by connection apply', () => {
    const affected = (profile: string, appliedProfile: null | string, hasExplicitProfileRoute = false) =>
      connectionApplyAffectsPoolProfile({
        appliedProfile,
        hasExplicitProfileRoute,
        primaryProfile: 'default',
        profile
      })

    expect(affected('default', 'default', true)).toBe(true)
    expect(affected('writer', 'writer', true)).toBe(true)
    expect(affected('writer', 'reviewer', true)).toBe(false)
    expect(affected('writer', null)).toBe(true)
    expect(affected('writer', null, true)).toBe(false)
  })

  it('does not reuse an incompatible cached pool route', () => {
    expect(shouldReuseBackendRoute('remote-global', 'local')).toBe(false)
    expect(shouldReuseBackendRoute('remote-profile', 'local')).toBe(false)
    expect(shouldReuseBackendRoute('local', 'remote-global')).toBe(false)
    expect(shouldReuseBackendRoute('remote-global', 'remote-global')).toBe(true)
    expect(shouldReuseBackendRoute('remote-profile', 'remote-profile')).toBe(true)
  })

  it('keeps runtime selection on applied config across save-only staging, then promotes on apply', () => {
    const persisted = {
      mode: 'remote',
      profiles: { writer: { mode: 'local' } }
    }

    const live = createAppliedConnectionConfig({ mode: 'remote', profiles: {} })

    const selection = () => {
      const config = live.current()

      return selectBackendSelection({
        explicitLocal: profileLocalOverride(config, 'writer'),
        explicitRemote: Boolean(profileRemoteOverride(config, 'writer')),
        globalRemote: modeIsRemoteLike(config.mode),
        primaryProfile: 'default',
        profile: 'writer'
      })
    }

    expect(persisted.profiles.writer.mode).toBe('local')
    expect(selection()).toEqual({ route: 'remote-global', target: 'pool' })

    live.promote(persisted)
    expect(selection()).toEqual({ route: 'local', target: 'pool' })
  })

  it('reuses a compatible replacement installed while incompatible teardown awaits', async () => {
    type Entry = { id: string; routeIdentity: BackendSelection['route'] }
    let current: Entry | undefined = { id: 'stale', routeIdentity: 'remote-global' }
    let releaseTeardown: (() => void) | undefined

    const teardownStarted = new Promise<void>(resolve => {
      releaseTeardown = resolve
    })

    let continueTeardown: (() => void) | undefined

    const teardownCanFinish = new Promise<void>(resolve => {
      continueTeardown = resolve
    })

    const create = vi.fn(() => {
      const entry: Entry = { id: 'replacement', routeIdentity: 'local' }
      current = entry

      return entry
    })

    const teardown = vi.fn(async (entry: Entry) => {
      if (current === entry) {
        current = undefined
      }

      releaseTeardown?.()
      await teardownCanFinish
    })

    const first = ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })
    await teardownStarted
    const second = await ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })
    continueTeardown?.()
    const firstResult = await first

    expect(firstResult).toBe(second)
    expect(current).toBe(second)
    expect(create).toHaveBeenCalledTimes(1)
    expect(teardown).toHaveBeenCalledTimes(1)
  })

  it('queues creation behind profile teardown instead of replacing during teardown', async () => {
    const queue = createProfileAsyncQueue()
    const events: string[] = []
    let releaseTeardown!: () => void

    const teardownDone = new Promise<void>(resolve => {
      releaseTeardown = resolve
    })

    const teardown = queue.run('writer', async () => {
      events.push('teardown:start')
      await teardownDone
      events.push('teardown:end')
    })

    const creation = queue.run('writer', () => {
      events.push('create')
    })

    await Promise.resolve()
    expect(events).toEqual(['teardown:start'])
    releaseTeardown()
    await Promise.all([teardown, creation])
    expect(events).toEqual(['teardown:start', 'teardown:end', 'create'])
  })

  it('tears down the promoted profile pool entry and the previous primary lifecycle', () => {
    expect(planProfilePromotion('default', 'writer')).toEqual({
      poolProfileToTeardown: 'writer',
      primaryProfileToTeardown: 'default'
    })
  })
})
