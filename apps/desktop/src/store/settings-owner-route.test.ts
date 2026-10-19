// @vitest-environment jsdom
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { capabilityScoped, profileScopeKey } from '@/api/client'
import type { HermesConnection } from '@/global'

// Native modules have their own compiler project. Import the actual runtime
// modules without pulling their non-strict graph into the renderer typecheck.
const nativeModules = {
  recycle: '../../electron/backend-recycle.ts',
  config: '../../electron/connection-config.ts',
  registry: '../../electron/connection-registry.ts'
}

const { recyclePinnedBackend } = await import(nativeModules.recycle)

const { resolveProfileBackendRoute, resolveRegistryApiConnection, resolveSettingsProfileConnection } = await import(
  nativeModules.config
)

const { resolveRegistryLocalRoute } = await import(nativeModules.registry)

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  ensureGatewayForAgent: vi.fn(),
  ensureGatewayForProfile: vi.fn(),
  openGatewayForProfile: vi.fn()
}))
vi.mock('@/hermes', () => ({ getProfiles: vi.fn(), setApiRequestProfile: vi.fn() }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import {
  $settingsOwner,
  setSettingsScope,
  settingsGatewayRequestOwnership,
  settingsRecycleScope
} from '@/store/settings-scope'

const descriptors = {
  default: {
    mode: 'local',
    authMode: 'token',
    baseUrl: 'http://127.0.0.1:31001',
    token: 'synthetic-primary',
    profile: 'default',
    connectionId: 'local'
  },
  research: {
    mode: 'local',
    authMode: 'token',
    baseUrl: 'http://127.0.0.1:31001',
    token: 'synthetic-primary',
    profile: 'research',
    connectionId: 'local'
  }
} as const

const originalBridge = window.hermesDesktop

afterEach(() => {
  $connection.set(null)
  setSettingsScope('default')
  window.hermesDesktop = originalBridge
  vi.clearAllMocks()
})

it.each(['default', 'research'] as const)(
  'captures the selected pooled descriptor from foreground %s, and rejects its replacement',
  async foreground => {
    const backend = vi.fn(async (_id: string, profile: 'default' | 'research') => descriptors[profile])
    window.hermesDesktop = {
      ...originalBridge,
      getConnectionFor: ({ connectionId, profile, expectedOwner }) =>
        resolveSettingsProfileConnection(profile, expectedOwner, (target: string) =>
          resolveRegistryApiConnection({ profile: target }, connectionId, backend)
        )
    }
    $activeGatewayProfile.set(foreground)
    $connection.set(descriptors[foreground] as HermesConnection)

    // A → B → A crosses the real store/API/native owner boundary, without
    // launching a backend. Only process acquisition is replaced by descriptors.
    for (const profile of [foreground, foreground === 'default' ? 'research' : 'default', foreground] as const) {
      setSettingsScope(profile)
      await vi.waitFor(() => expect($settingsOwner.get()?.profile).toBe(profile))
      const owner = $settingsOwner.get()!

      expect(owner.connectionOwner?.baseUrl).toBe(descriptors[profile].baseUrl)
      expect(resolveRegistryLocalRoute(profile, {}).delegate).toBe(true)
      expect(resolveProfileBackendRoute(profile, { primaryProfile: 'default' }).backend).toBe('primary')

      for (const [method, path] of [
        ['GET', '/api/config'],
        ['PUT', '/api/config'],
        ['POST', '/api/providers/oauth/openai-codex/start']
      ]) {
        const request = { ...capabilityScoped(owner), method, path }
        await expect(resolveRegistryApiConnection(request, 'local', backend)).resolves.toEqual(descriptors[profile])
        await expect(
          resolveRegistryApiConnection(request, 'local', async () => ({ ...descriptors[profile], token: 'replaced' }))
        ).rejects.toThrow('Backend changed')
      }

      const primaryPromise = Promise.resolve(descriptors.default)
      const teardownPool = vi.fn()
      const teardownPrimary = vi.fn()
      await recyclePinnedBackend(owner, {
        registry: {
          version: 2,
          launchMode: 'primary',
          lastUsed: 'local',
          primary: 'local',
          connections: [{ id: 'local', kind: 'local', label: 'This device' }]
        },
        routeOptions: { primaryProfile: 'default' },
        primarySshKey: '',
        effectiveSshFingerprint: vi.fn(),
        primaryPromise: () => primaryPromise,
        pool: new Map([['research', { connectionPromise: Promise.resolve(descriptors.research) }]]),
        sshState: () => undefined,
        teardownSsh: vi.fn(),
        teardownPool,
        teardownPrimary,
        notifyApplied: vi.fn()
      })
      expect(teardownPrimary).toHaveBeenCalledOnce()
      expect(teardownPool).not.toHaveBeenCalled()
    }
  }
)

it('validates the foreground source while routing a distinct selected-profile descriptor', async () => {
  const source = descriptors.default

  const target = {
    ...descriptors.research,
    baseUrl: 'http://127.0.0.1:32002',
    token: 'synthetic-research'
  }

  const backend = vi.fn(async (_id: string, profile: string) => (profile === 'research' ? target : source))
  window.hermesDesktop = {
    ...originalBridge,
    getConnectionFor: ({ connectionId, profile, expectedOwner }) =>
      resolveSettingsProfileConnection(profile, expectedOwner, (selected: string) =>
        resolveRegistryApiConnection({ profile: selected }, connectionId, backend)
      )
  }
  $activeGatewayProfile.set('default')
  $connection.set(source as HermesConnection)
  setSettingsScope('research')

  await vi.waitFor(() => expect($settingsOwner.get()?.profile).toBe('research'))
  const owner = $settingsOwner.get()!
  const request = { ...capabilityScoped(owner), method: 'PUT', path: '/api/config' }

  expect(owner.connectionOwner).toMatchObject({ baseUrl: source.baseUrl, token: source.token })
  expect(owner.connectionOwnerProfile).toBe('default')
  expect(settingsGatewayRequestOwnership(owner)).toEqual({
    expectedConnectionOwner: owner.cacheOwner,
    expectedOwner: { profile: 'default', connectionOwner: owner.connectionOwner }
  })
  expect(profileScopeKey(owner)).not.toBe(profileScopeKey({ ...owner, cacheOwner: owner.connectionOwner }))
  backend.mockClear()
  await expect(resolveRegistryApiConnection(request, 'local', backend)).resolves.toEqual(target)
  expect(backend.mock.calls).toEqual([
    ['local', 'default', undefined],
    ['local', 'research', undefined],
    ['local', 'default', undefined]
  ])
})

it('uses the captured legacy descriptor for gateway owner validation', () => {
  const legacy = {
    mode: 'remote',
    baseUrl: 'https://legacy.example',
    token: 'synthetic-legacy',
    profile: 'worker'
  } as const

  $activeGatewayProfile.set('worker')
  $connection.set(legacy as HermesConnection)

  expect(settingsGatewayRequestOwnership($settingsOwner.get()!)).toEqual({
    expectedConnectionOwner: expect.objectContaining({ baseUrl: legacy.baseUrl, token: legacy.token }),
    expectedOwner: {
      profile: 'worker',
      connectionOwner: expect.objectContaining({ baseUrl: legacy.baseUrl, token: legacy.token })
    }
  })
})

it('recycles the selected profile descriptor rather than the different foreground owner', async () => {
  const source = descriptors.default

  const target = {
    ...descriptors.research,
    baseUrl: 'http://127.0.0.1:32002',
    token: 'synthetic-research'
  }

  const primaryPromise = Promise.resolve(source)
  const selectedPromise = Promise.resolve(target)
  const teardownPool = vi.fn()

  window.hermesDesktop = {
    ...originalBridge,
    getConnectionFor: vi.fn(async ({ profile }) =>
      (profile === 'research' ? target : source) as HermesConnection
    ) as NonNullable<NonNullable<typeof window.hermesDesktop>['getConnectionFor']>
  }
  $activeGatewayProfile.set('default')
  $connection.set(source as HermesConnection)
  setSettingsScope('research')

  await vi.waitFor(() =>
    expect($settingsOwner.get()?.cacheOwner).toMatchObject({ baseUrl: target.baseUrl, token: target.token })
  )
  const scope = settingsRecycleScope($settingsOwner.get()!)

  expect(scope).toMatchObject({
    connectionId: 'local',
    connectionOwner: { baseUrl: target.baseUrl, token: target.token },
    profile: 'research'
  })
  await expect(
    recyclePinnedBackend(scope!, {
      registry: {
        version: 2,
        launchMode: 'primary',
        lastUsed: 'local',
        primary: 'local',
        connections: [{ id: 'local', kind: 'local', label: 'This device' }]
      },
      routeOptions: { primaryProfile: 'default', profileRemoteOverride: true },
      primarySshKey: '',
      effectiveSshFingerprint: vi.fn(),
      primaryPromise: () => primaryPromise,
      pool: new Map([['research', { connectionPromise: selectedPromise }]]),
      sshState: () => undefined,
      teardownSsh: vi.fn(),
      teardownPool,
      teardownPrimary: vi.fn(),
      notifyApplied: vi.fn()
    })
  ).resolves.toBe('pool')
  expect(teardownPool).toHaveBeenCalledWith('research')
})

it('does not adopt a delayed descriptor after the selected gateway changes or acquisition fails', async () => {
  let finish!: (connection: HermesConnection) => void
  window.hermesDesktop = {
    ...originalBridge,
    getConnectionFor: vi.fn(
      () =>
        new Promise<HermesConnection>(resolve => {
          finish = resolve
        })
    )
  }
  $activeGatewayProfile.set('default')
  $connection.set(descriptors.default as HermesConnection)
  setSettingsScope('research')
  expect($settingsOwner.get()).toBeNull()

  const replacement = {
    ...descriptors.default,
    connectionId: 'replacement',
    baseUrl: 'https://replacement.example',
    mode: 'remote'
  } as HermesConnection

  const finishOld = finish
  window.hermesDesktop.getConnectionFor = vi.fn().mockRejectedValue(new Error('unavailable'))
  $connection.set(replacement)
  finishOld(descriptors.research as HermesConnection)
  await Promise.resolve()
  await Promise.resolve()
  expect($settingsOwner.get()).toBeNull()
  await vi.waitFor(() => expect(notifyError).toHaveBeenCalledOnce())
  const refresh = vi.mocked(notifyError).mock.calls[0][2]!.action!.onClick
  const retry = vi.fn().mockResolvedValue({ ...replacement, profile: 'research' })
  window.hermesDesktop.getConnectionFor = retry
  refresh()
  await vi.waitFor(() => expect($settingsOwner.get()?.profile).toBe('research'))
  expect($settingsOwner.get()?.connectionOwner?.baseUrl).toBe(replacement.baseUrl)
  setSettingsScope('default')
  refresh()
  expect(retry).toHaveBeenCalledOnce()
  expect($settingsOwner.get()?.connectionOwner?.baseUrl).toBe(replacement.baseUrl)
})

it.each(['before', 'during'])('rejects same-id replacement %s selected-profile acquisition', async when => {
  let source = descriptors.default as HermesConnection
  const expectedOwner = { profile: 'default', connectionOwner: { ...source } }

  const resolve = vi.fn(async (profile: string) => {
    if (profile === 'research') {
      if (when === 'during') {
        source = { ...source, token: 'replacement' }
      }

      return descriptors.research
    }

    return source
  })

  if (when === 'before') {
    source = { ...source, token: 'replacement' }
  }

  await expect(resolveSettingsProfileConnection('research', expectedOwner, resolve)).rejects.toThrow('Backend changed')

  if (when === 'before') {
    expect(resolve).not.toHaveBeenCalledWith('research')
  }
})
