import { setImmediate } from 'node:timers/promises'

import { describe, expect, it, vi } from 'vitest'

import { BackendDialClaims } from './backend-dial-claim'
import type { ProfileRouteOptions, RegistryBackendRequestScope } from './connection-config'
import { backendScopeKey } from './connection-registry'
import { DEFAULT_FETCH_TIMEOUT_MS } from './hardening'
import { LocalBackendSpawnCoordinator, type LocalBackendSpawnRequest } from './pool-spawn-coordinator'
import { createRegistryApiDispatcher, type RegistryApiRequest } from './registry-api-dispatch'

interface TestBackend extends RegistryBackendRequestScope {
  baseUrl: string
  profile: string
}

// The primary is already running; both background slots are occupied. A cold
// profile dial must really queue, not succeed because the test mocked the cap.
async function saturatedLocalRuntime(options: ProfileRouteOptions = {}) {
  const coordinator = new LocalBackendSpawnCoordinator(3)

  const releases = await Promise.all(
    ['busy-a', 'busy-b'].map(key => coordinator.request(key, { priority: 'background' }).acquired)
  )

  const queued: LocalBackendSpawnRequest[] = []
  const claims = new BackendDialClaims()
  const primary: TestBackend = { baseUrl: 'http://localhost:9100', profile: 'primary' }
  const warm = new Map<string, TestBackend>([[primary.profile, primary]])

  const ensureRegistryBackend = vi.fn(
    async (connectionId: string, profile?: null | string, correlation?: string, opts?: { passive?: boolean }) => {
      expect(correlation).toBe(opts?.passive ? '' : undefined)
      const key = backendScopeKey(connectionId, profile || (connectionId === 'local' ? primary.profile : 'default'))
      const existing = warm.get(key)

      if (existing) {
        return existing
      }

      if (opts?.passive) {
        throw new Error('No warm backend')
      }

      expect(connectionId).toBe('local')
      const request = coordinator.request(key, { priority: 'background' })
      queued.push(request)
      releases.push(await request.acquired)
      const backend = { baseUrl: 'http://localhost:9101', profile: profile || 'default' }
      warm.set(key, backend)

      return backend
    }
  )

  const fetchJsonForBackend = vi.fn(async (connection: TestBackend, path: string, opts) => {
    expect(Object.keys(opts).sort()).toEqual(['body', 'method', 'timeoutMs', 'upload'])
    expect(opts.timeoutMs).toBeGreaterThan(0)
    const url = new URL(path, connection.baseUrl)
    // Like profile-aware handlers, an explicit query overrides the process's
    // launch home. A dropped query therefore returns another profile's data.
    const profile = url.searchParams.get('profile') || connection.profile

    return url.pathname === '/api/sessions'
      ? { sessions: [{ id: 'session-1', profile }] }
      : { gateway: url.hostname, profile }
  })

  const dispatch = createRegistryApiDispatcher({
    backendDialClaims: claims,
    ensureRegistryBackend,
    fetchJsonForBackend,
    profileRouteOptions: (_profile, request) => ({
      primaryProfile: primary.profile,
      globalRemote: false,
      primaryRemoteActive: false,
      profileRemoteOverride: false,
      ...options,
      requestMethod: request.method
    })
  })

  return {
    claims,
    coordinator,
    dispatch,
    ensureRegistryBackend,
    fetchJsonForBackend,
    primary,
    releases,
    warm,
    async close() {
      for (const request of queued) {
        request.cancel()
      }

      for (const release of releases) {
        release()
      }

      await setImmediate()
    }
  }
}

interface ReuseCase {
  name: string
  request: RegistryApiRequest
  expectedPath: string
  expectedProfile: string
  pendingProfile?: string
}

const reuseCases: ReuseCase[] = [
  ...['oauth', 'custom-endpoints'].map(endpoint => ({
    name: `provider ${endpoint} list`,
    request: { path: `/api/providers/${endpoint}`, profile: 'work', method: 'GET' },
    expectedPath: `/api/providers/${endpoint}?profile=work`,
    expectedProfile: 'work'
  })),
  {
    name: 'config',
    request: { path: '/api/config', profile: 'work' },
    expectedPath: '/api/config?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'schema',
    request: { path: '/api/config/schema', profile: 'work' },
    expectedPath: '/api/config/schema?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'raw config',
    request: { path: '/api/config/raw', profile: 'work' },
    expectedPath: '/api/config/raw?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'query flags',
    request: { path: '/api/config?reveal=&scope=full#section', profile: 'work', method: 'get' },
    expectedPath: '/api/config?reveal=&scope=full&profile=work#section',
    expectedProfile: 'work'
  },
  {
    name: 'explicit cross-profile selector',
    request: { path: '/api/config?profile=other&reveal=true', profile: 'work' },
    expectedPath: '/api/config?profile=other&reveal=true',
    expectedProfile: 'other'
  },
  {
    name: 'explicit empty selector',
    request: { path: '/api/config?profile=&reveal=false', profile: 'work' },
    expectedPath: '/api/config?profile=&reveal=false',
    expectedProfile: 'primary'
  },
  {
    name: 'profile-aware write',
    request: { path: '/api/config', profile: 'work', method: 'PUT', body: { model: 'model-a' }, timeoutMs: 1234 },
    expectedPath: '/api/config?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'missing profile keeps the window primary',
    request: { path: '/api/config' },
    expectedPath: '/api/config',
    expectedProfile: 'primary'
  },
  {
    name: 'default is explicit even with a named primary',
    request: { path: '/api/config', profile: 'default' },
    expectedPath: '/api/config?profile=default',
    expectedProfile: 'default'
  },
  {
    name: 'pending requested-profile claim cannot capture the primary route',
    request: { path: '/api/config', profile: 'work' },
    expectedPath: '/api/config?profile=work',
    expectedProfile: 'work',
    pendingProfile: 'work'
  },
  {
    name: 'pending primary claim is shared by the selected backend',
    request: { path: '/api/config', profile: 'work' },
    expectedPath: '/api/config?profile=work',
    expectedProfile: 'work',
    pendingProfile: 'primary'
  },
  {
    name: 'passive read never joins a dial claim',
    request: { path: '/api/config', profile: 'work', passive: true },
    expectedPath: '/api/config?profile=work',
    expectedProfile: 'work',
    pendingProfile: 'primary'
  }
]

interface BoundaryCase {
  name: string
  request: RegistryApiRequest
  connectionId?: string
  options?: ProfileRouteOptions
  backend?: TestBackend
  expectedDial: null | string | undefined
  expectedPath: string
  expectedProfile: string
  routeProfile?: null | string
  requestProfile?: null | string
  cold?: boolean
  rejects?: boolean
}

const remote: TestBackend = { baseUrl: 'https://gateway.example.test', profile: 'default', sharedRemote: true }
const ssh: TestBackend = { baseUrl: 'http://localhost:9102', profile: 'default', remoteProfile: 'default' }

const boundaryCases: BoundaryCase[] = [
  ...[
    ...['oauth', 'custom-endpoints'].flatMap(endpoint =>
      ['DELETE', 'PATCH', 'POST', 'PUT'].map(method => ({ path: `/api/providers/${endpoint}`, method }))
    ),
    { path: '/api/providers/oauth/nous', method: 'DELETE' },
    { path: '/api/providers/oauth/nous/start', method: 'POST' },
    { path: '/api/providers/oauth/nous/submit', method: 'POST' },
    { path: '/api/providers/oauth/nous/poll/session-1', method: 'GET' },
    { path: '/api/providers/oauth/sessions/session-1', method: 'DELETE' },
    { path: '/api/providers/custom-endpoints/endpoint-1/activate', method: 'POST' },
    { path: '/api/providers/custom-endpoints/endpoint-1', method: 'DELETE' }
  ].map(({ path, method }) => ({
    name: `${method} ${path} remains process-scoped`,
    request: { path, method, profile: 'work' },
    expectedDial: 'work',
    expectedPath: path,
    expectedProfile: 'work',
    cold: true
  })),
  ...['DELETE', 'PATCH', 'POST'].map(method => ({
    name: `${method} config is not allowlisted`,
    request: { path: '/api/config', profile: 'work', method, body: { value: 'synthetic' } },
    expectedDial: 'work',
    expectedPath: '/api/config',
    expectedProfile: 'work',
    cold: true
  })),
  {
    name: 'unknown destructive path',
    request: { path: '/api/memory/reset?confirm=true', profile: 'work', method: 'POST' },
    expectedDial: 'work',
    expectedPath: '/api/memory/reset?confirm=true',
    expectedProfile: 'work',
    cold: true
  },
  {
    name: 'session writes remain process-scoped',
    request: { path: '/api/sessions/session-1', profile: 'work', method: 'DELETE' },
    expectedDial: 'work',
    expectedPath: '/api/sessions/session-1',
    expectedProfile: 'work',
    cold: true
  },
  {
    name: 'global remote primary cannot serve explicit local',
    request: { path: '/api/config', profile: 'work' },
    options: { globalRemote: true, primaryRemoteActive: true },
    expectedDial: 'work',
    expectedPath: '/api/config',
    expectedProfile: 'work',
    cold: true
  },
  {
    name: 'per-profile remote primary cannot serve explicit local',
    request: { path: '/api/config', profile: 'work' },
    options: { primaryRemoteActive: true, ownEntry: true },
    expectedDial: 'work',
    expectedPath: '/api/config',
    expectedProfile: 'work',
    cold: true
  },
  {
    name: 'target SSH override retains its namespace',
    request: { path: '/api/config?profile=work&reveal=true', profile: 'work' },
    options: { profileRemoteOverride: true, backendProfile: 'default' },
    backend: ssh,
    expectedDial: 'work',
    expectedPath: '/api/config?profile=default&reveal=true',
    expectedProfile: 'default'
  },
  {
    name: 'target URL override remains on its gateway',
    request: { path: '/api/config', profile: 'work' },
    options: { profileRemoteOverride: true },
    backend: { ...remote, profile: 'work', sharedRemote: false },
    expectedDial: 'work',
    expectedPath: '/api/config',
    expectedProfile: 'work'
  },
  {
    name: 'registry URL scopes the requested profile',
    connectionId: 'remote-a',
    request: { path: '/api/config?reveal=true', profile: 'work' },
    backend: remote,
    expectedDial: 'work',
    expectedPath: '/api/config?reveal=true&profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'registry missing profile retains the gateway default',
    connectionId: 'remote-a',
    request: { path: '/api/config' },
    backend: remote,
    expectedDial: undefined,
    expectedPath: '/api/config',
    expectedProfile: 'default'
  },
  {
    name: 'registry SSH translates only self selectors',
    connectionId: 'ssh-a',
    request: { path: '/api/profiles/sessions/sidebar?profile=all&recents_profile=work&limit=10', profile: 'work' },
    backend: ssh,
    expectedDial: 'work',
    expectedPath: '/api/profiles/sessions/sidebar?profile=all&recents_profile=default&limit=10',
    expectedProfile: 'all'
  },
  {
    name: 'local deletion separates route and target',
    request: { path: '/api/profiles/work', profile: 'work', method: 'DELETE' },
    routeProfile: null,
    requestProfile: 'work',
    expectedDial: 'primary',
    expectedPath: '/api/profiles/work',
    expectedProfile: 'primary'
  },
  {
    name: 'remote deletion separates route and target',
    connectionId: 'remote-a',
    request: { path: '/api/profiles/work', profile: 'work', method: 'DELETE' },
    routeProfile: null,
    requestProfile: 'work',
    backend: remote,
    expectedDial: null,
    expectedPath: '/api/profiles/work?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'SSH deletion translates target rather than route',
    connectionId: 'ssh-a',
    request: { path: '/api/profiles/work?profile=work', profile: 'work', method: 'DELETE' },
    routeProfile: null,
    requestProfile: 'work',
    backend: ssh,
    expectedDial: null,
    expectedPath: '/api/profiles/work?profile=default',
    expectedProfile: 'default'
  },
  {
    name: 'warm passive isolated route',
    request: { path: '/api/status', profile: 'work', passive: true },
    backend: { baseUrl: 'http://localhost:9101', profile: 'work' },
    expectedDial: 'work',
    expectedPath: '/api/status',
    expectedProfile: 'work'
  },
  {
    name: 'cold passive isolated route refuses without claiming or queueing',
    request: { path: '/api/status', profile: 'work', passive: true },
    expectedDial: 'work',
    expectedPath: '/api/status',
    expectedProfile: 'work',
    rejects: true
  },
  {
    name: 'upload passes through an isolated route',
    request: {
      path: '/api/files/upload',
      profile: 'work',
      method: 'POST',
      upload: { name: 'sample.txt', data: 'synthetic' }
    },
    backend: { baseUrl: 'http://localhost:9101', profile: 'work' },
    expectedDial: 'work',
    expectedPath: '/api/files/upload',
    expectedProfile: 'work'
  },
  {
    name: 'local session reads carry registry ownership',
    request: { path: '/api/sessions', profile: 'work' },
    expectedDial: 'primary',
    expectedPath: '/api/sessions?profile=work',
    expectedProfile: 'work'
  },
  {
    name: 'remote session reads carry registry ownership',
    connectionId: 'remote-a',
    request: { path: '/api/sessions', profile: 'work' },
    backend: remote,
    expectedDial: 'work',
    expectedPath: '/api/sessions?profile=work',
    expectedProfile: 'work'
  }
]

describe('registry REST dispatch', () => {
  it.each(reuseCases)('reuses the local primary without spawning: $name', async test => {
    const runtime = await saturatedLocalRuntime()
    let releaseClaim: (() => void) | undefined

    const pending = test.pendingProfile
      ? runtime.claims.run(
          test.pendingProfile,
          () =>
            new Promise<TestBackend>(resolve => {
              releaseClaim = () => resolve(runtime.primary)
            })
        )
      : undefined

    const result = runtime.dispatch(test.request, 'local')

    // Observe settlement without waiting for a slot timeout; attach the error
    // branch before cleanup cancels the queued dial on the unfixed code.
    const settled = result.then(
      value => ({ value }),
      error => ({ error })
    )

    try {
      await setImmediate()
      expect(runtime.coordinator.activeCount).toBe(2)
      expect(runtime.coordinator.queuedCount).toBe(0)

      if (test.pendingProfile === 'primary' && !test.request.passive) {
        expect(runtime.ensureRegistryBackend).not.toHaveBeenCalled()
        expect(runtime.fetchJsonForBackend).not.toHaveBeenCalled()
        releaseClaim!()
      } else {
        expect(runtime.ensureRegistryBackend.mock.calls).toEqual([
          test.request.passive ? ['local', 'primary', '', { passive: true }] : ['local', 'primary']
        ])
      }

      expect(await settled).toEqual({ value: { gateway: 'localhost', profile: test.expectedProfile } })
      expect(runtime.fetchJsonForBackend).toHaveBeenCalledWith(runtime.primary, test.expectedPath, {
        method: test.request.method,
        body: test.request.body,
        upload: test.request.upload,
        timeoutMs: test.request.timeoutMs || DEFAULT_FETCH_TIMEOUT_MS
      })

      if (test.pendingProfile && (test.pendingProfile !== 'primary' || test.request.passive)) {
        expect(runtime.claims.inFlight(test.pendingProfile)).toBe(true)
      }
    } finally {
      releaseClaim?.()
      await runtime.close()
      await Promise.all([settled, pending])
    }
  })

  it.each(boundaryCases)('preserves backend and request scope: $name', async test => {
    const runtime = await saturatedLocalRuntime(test.options)
    const connectionId = test.connectionId || 'local'
    const routeProfile = 'routeProfile' in test ? test.routeProfile : test.request.profile
    const requestProfile = 'requestProfile' in test ? test.requestProfile : test.request.profile
    const claimKey = backendScopeKey(connectionId, test.expectedDial)

    const registryKey = backendScopeKey(
      connectionId,
      test.expectedDial || (connectionId === 'local' ? 'primary' : 'default')
    )

    if (test.backend) {
      runtime.warm.set(registryKey, test.backend)
    }

    const result = runtime.dispatch(test.request, connectionId, routeProfile, requestProfile)

    const settled = result.then(
      value => ({ value }),
      error => ({ error })
    )

    // A same-named profile on a second gateway is a different backend and must
    // neither join this claim nor inherit its profile namespace or payload.
    const siblingId = 'remote-b'
    const siblingKey = backendScopeKey(siblingId, test.request.profile)
    runtime.warm.set(siblingKey, { ...remote, baseUrl: 'https://other-gateway.example.test' })
    const sibling = runtime.dispatch({ path: '/api/config', profile: test.request.profile }, siblingId)

    try {
      expect(runtime.claims.inFlight(claimKey)).toBe(!test.request.passive)
      expect(await sibling).toEqual({
        gateway: 'other-gateway.example.test',
        profile: test.request.profile || 'default'
      })
      await setImmediate()
      expect(runtime.coordinator.activeCount).toBe(2)
      expect(runtime.coordinator.queuedCount).toBe(test.cold ? 1 : 0)
      expect(runtime.ensureRegistryBackend.mock.calls[0]).toEqual(
        test.request.passive
          ? [connectionId, test.expectedDial, '', { passive: true }]
          : [connectionId, test.expectedDial]
      )

      if (test.cold) {
        runtime.releases[0]()
      }

      if (test.rejects) {
        expect(await settled).toEqual({ error: new Error('No warm backend') })
        expect(runtime.fetchJsonForBackend).toHaveBeenCalledTimes(1) // sibling only
      } else {
        const value =
          test.request.path === '/api/sessions'
            ? { sessions: [{ id: 'session-1', profile: test.expectedProfile, connection_id: connectionId }] }
            : { gateway: new URL((test.backend || runtime.primary).baseUrl).hostname, profile: test.expectedProfile }

        expect(await settled).toEqual({ value })
        expect(runtime.fetchJsonForBackend).toHaveBeenCalledWith(runtime.warm.get(registryKey), test.expectedPath, {
          method: test.request.method,
          body: test.request.body,
          upload: test.request.upload,
          timeoutMs: test.request.timeoutMs || DEFAULT_FETCH_TIMEOUT_MS
        })
      }

      expect(runtime.claims.inFlight(claimKey)).toBe(false)
    } finally {
      await runtime.close()
      await Promise.all([settled, sibling])
    }
  })
})
