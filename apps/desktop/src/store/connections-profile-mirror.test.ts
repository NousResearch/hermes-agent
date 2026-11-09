import { atom } from 'nanostores'
import { expect, it, vi } from 'vitest'

// Main owns each connection's last-used profile (it boots a remote primary
// before any window exists). Windows do not see each other's picks, so this
// window must import its earlier record once and then report only the entries
// it changes, never its whole stale copy.

const $activeGatewayProfile = atom('default')

const $connection = atom<null | {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  registryScoped?: boolean
}>(null)

vi.mock('@/store/session', () => ({
  $connection,
  $activeSessionId: atom(null),
  $selectedStoredSessionId: atom(null),
  setActiveSessionId: vi.fn(),
  setSelectedStoredSessionId: vi.fn()
}))
vi.mock('@/store/project-scope', () => ({ exitProjectScope: vi.fn() }))
vi.mock('@/store/gateway-switch', () => ({ supersedeGatewaySwitch: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $freshSessionRequest: atom(0),
  $newChatProfile: atom(null),
  $newChatRoute: atom(null),
  $showAllProfiles: atom(false),
  captureNewChatSource: vi.fn(),
  currentNewChatIntent: () => 0,
  ensureGatewayAgent: vi.fn(async () => undefined),
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  openGatewayAgent: vi.fn(async () => undefined),
  refreshActiveProfile: vi.fn(async () => undefined),
  requestFreshSession: vi.fn()
}))

it('reports single profile changes to main and never pushes its whole map', async () => {
  localStorage.setItem('hermes.desktop.lastProfileByConnection', JSON.stringify({ homelab: 'work', 'work-vps': 'ops' }))

  const profile = {
    importConnections: vi.fn(async () => undefined),
    rememberConnection: vi.fn(async () => undefined),
    // The whole-map push this replaces: an older window erased newer picks.
    rememberConnections: vi.fn(async () => undefined)
  }

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { profile }

  const { _resetConnectionsForTests, initializeConnectionsRegistry } = await import('./connections')

  // Importing the module touches no bridge; boot imports the pre-mirror record once.
  expect(profile.importConnections).not.toHaveBeenCalled()
  void initializeConnectionsRegistry().catch(() => undefined)
  expect(profile.importConnections).toHaveBeenCalledTimes(1)
  expect(profile.importConnections).toHaveBeenCalledWith({ homelab: 'work', 'work-vps': 'ops' })

  $connection.set({ connectionId: 'homelab', mode: 'remote', profile: 'marina', registryScoped: true })
  $activeGatewayProfile.set('marina')

  expect(profile.rememberConnection).toHaveBeenCalledTimes(1)
  expect(profile.rememberConnection).toHaveBeenCalledWith('homelab', 'marina')

  // Clearing this window's copy is not a change main should apply.
  _resetConnectionsForTests()

  expect(profile.rememberConnection).toHaveBeenCalledTimes(1)
  expect(profile.rememberConnections).not.toHaveBeenCalled()

  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})
