import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const $activeGatewayProfile = atom('default')
const $newChatProfile = atom<string | null>(null)
const $newChatRoute = atom<{ connectionId: string; profile: string } | null>(null)
const $freshSessionRequest = atom(0)
const $showAllProfiles = atom(false)
const $connection = atom<{ connectionId: string; mode: 'remote'; profile: string; registryScoped: true } | null>(null)
const $activeSessionId = atom<string | null>(null)
const $selectedStoredSessionId = atom<string | null>(null)
const $defaultProfileRoute = atom<{ connectionId: string; profile: string } | null>(null)
const getProfiles = vi.fn()
let newChatIntentRevision = 0

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
  $newChatRoute,
  $freshSessionRequest,
  $showAllProfiles,
  normalizeProfileKey: (value: string | null | undefined) => value?.trim() || 'default',
  ensureGatewayAgent,
  openGatewayAgent,
  currentNewChatIntent: () => newChatIntentRevision,
  captureNewChatSource: vi.fn(() => {
    newChatIntentRevision += 1
  }),
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

const { initializeConnectionsRegistry, _resetConnectionsForTests, selectConnection, setConnectionsRegistry } =
  await import('./connections')

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
  $showAllProfiles.set(false)
  $newChatRoute.set(null)
  newChatIntentRevision = 0
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

  it('keeps All profiles browse mode when correcting a stale profile during boot', async () => {
    $showAllProfiles.set(true)
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'marina' }] })

    await initializeConnectionsRegistry()

    expect(ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default', expect.anything())
    expect($showAllProfiles.get()).toBe(true)
  })

  it('keeps a pinned new-chat route when boot corrects the active profile', async () => {
    $newChatRoute.set({ connectionId: 'homelab', profile: 'marina' })
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'marina' }] })

    await initializeConnectionsRegistry()

    expect($activeGatewayProfile.get()).toBe('default')
    expect($newChatRoute.get()).toEqual({ connectionId: 'homelab', profile: 'marina' })
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

  it('keeps All profiles when an explicit default already points at the booted route', async () => {
    $showAllProfiles.set(true)
    $defaultProfileRoute.set({ connectionId: 'homelab', profile: 'office-evals-windows' })

    await initializeConnectionsRegistry()

    expect($showAllProfiles.get()).toBe(true)
  })

  it('still leaves All profiles on an intentional user connection pick', async () => {
    setConnectionsRegistry(registry as Parameters<typeof setConnectionsRegistry>[0])
    $showAllProfiles.set(true)
    $newChatRoute.set({ connectionId: 'homelab', profile: 'office-evals-windows' })

    await selectConnection('homelab', { profile: 'marina' })

    expect($showAllProfiles.get()).toBe(false)
    expect($newChatRoute.get()).toBe(null)
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
