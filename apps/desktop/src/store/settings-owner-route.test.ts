// @vitest-environment jsdom
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { capabilityScoped } from '@/api/client'
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
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner, setSettingsScope } from '@/store/settings-scope'

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
