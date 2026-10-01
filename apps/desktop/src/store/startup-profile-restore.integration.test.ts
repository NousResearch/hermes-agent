import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const $activeGatewayProfile = atom('default')
const $newChatProfile = atom<string | null>(null)
const $freshSessionRequest = atom(0)
const $showAllProfiles = atom(false)
const $connection = atom<{ connectionId: string; mode: 'remote'; profile: string; registryScoped: true } | null>(null)
const $activeSessionId = atom<string | null>(null)
const $selectedStoredSessionId = atom<string | null>(null)
const $defaultProfileRoute = atom<{ connectionId: string; profile: string } | null>(null)
const getProfiles = vi.fn()

const ensureGatewayAgent = vi.fn(
  async (connectionId: string, profile: string, options?: { beforeActivate?: () => boolean }) => {
    if (options?.beforeActivate?.() === false) {
      return
    }

    $activeGatewayProfile.set(profile)
    $connection.set({ connectionId, mode: 'remote', profile, registryScoped: true })
  }
)

const openGatewayAgent = vi.fn(async () => undefined)
const setLastUsed = vi.fn(async () => ({ registry: { connections: [], primary: 'homelab' } }))

vi.mock('@/api/profiles', () => ({ getProfiles }))
vi.mock('@/store/session', () => ({ $connection, $activeSessionId, $selectedStoredSessionId }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $newChatProfile,
  $freshSessionRequest,
  $showAllProfiles,
  normalizeProfileKey: (value: string | null | undefined) => value?.trim() || 'default',
  ensureGatewayAgent,
  openGatewayAgent,
  captureNewChatSource: vi.fn(),
  refreshActiveProfile: vi.fn(async () => undefined),
  requestFreshSession: vi.fn()
}))
vi.mock('@/store/gateway-switch', () => ({
  beginGatewaySwitch: () => 1,
  endGatewaySwitch: vi.fn(),
  recoverActiveSourceAfterFailedGatewaySwitch: vi.fn()
}))
vi.mock('@/store/default-profile', () => ({
  $defaultProfileRoute,
  refreshDefaultProfile: vi.fn(async () => $defaultProfileRoute.get())
}))

const { initializeConnectionsRegistry, _resetConnectionsForTests } = await import('./connections')

const registry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device' },
    { id: 'homelab', kind: 'remote', label: 'Homelab' }
  ],
  primary: 'homelab',
  lastUsed: 'homelab',
  launchMode: 'primary',
  version: 2
}

beforeEach(() => {
  localStorage.clear()
  _resetConnectionsForTests()
  $activeGatewayProfile.set('office-evals-windows')
  $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'office-evals-windows', registryScoped: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $defaultProfileRoute.set(null)
  ensureGatewayAgent.mockClear()
  openGatewayAgent.mockClear()
  getProfiles.mockReset()
  setLastUsed.mockClear()
  vi.stubGlobal('window', {
    hermesDesktop: { connections: { list: async () => registry, setLastUsed } },
    localStorage,
    location: window.location
  })
})

describe('restoring a remote primary without an explicit default', () => {
  it('re-homes a Windows-only legacy profile onto the remote default', async () => {
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'marina' }] })

    await initializeConnectionsRegistry()

    expect(getProfiles).toHaveBeenCalledWith({ connectionId: 'homelab' })
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('preserves a matching remote profile, without any re-home', async () => {
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'office-evals-windows' }] })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('office-evals-windows')
  })

  it('does not infer absence or switch profiles when the remote roster is unavailable', async () => {
    getProfiles.mockRejectedValue(new Error('offline'))

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('office-evals-windows')
  })

  it('honors an explicit default route without substituting the legacy profile', async () => {
    $defaultProfileRoute.set({ connectionId: 'homelab', profile: 'marina' })

    await initializeConnectionsRegistry()

    expect(getProfiles).not.toHaveBeenCalled()
    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'marina', expect.anything())
  })

  it('does not undo a user profile choice made while the roster is loading', async () => {
    let answer: (result: { profiles: { name: string }[] }) => void = () => undefined
    getProfiles.mockImplementation(
      () =>
        new Promise(resolve => {
          answer = resolve
        })
    )

    const restoring = initializeConnectionsRegistry()
    await vi.waitFor(() => expect(getProfiles).toHaveBeenCalled())
    $activeGatewayProfile.set('marina')
    answer({ profiles: [{ name: 'default' }, { name: 'marina' }] })
    await restoring

    expect(ensureGatewayAgent).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe('marina')
  })
})
