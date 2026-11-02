import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster, DesktopConnectionsRegistry } from '@/global'

import { deferred } from '../test/deferred'

// The launch restore onto the last-used gateway dials while the window already
// shows the source main opened first, so the person can open a chat before it
// lands. A chat they opened is theirs: the restore gives way. A chat selected
// on its own (the cold-start route resume) is not, and the restore still lands.

const $activeGatewayProfile = atom('default')
const $freshSessionRequest = atom(0)
const $activeSessionId = atom<null | string>(null)
const $selectedStoredSessionId = atom<null | string>(null)

const $connection = atom<null | {
  connectionId?: string
  mode?: 'local' | 'remote'
  profile?: string
  registryScoped?: boolean
}>(null)

const ensureGatewayAgent = vi.fn(
  async (connectionId: null | string, profile: string, options?: { beforeActivate?: () => boolean }) => {
    if (options?.beforeActivate && !options.beforeActivate()) {
      return
    }

    $connection.set({ connectionId: connectionId ?? undefined, mode: 'remote', profile, registryScoped: true })
  }
)

const openGatewayAgent = vi.fn(async (_connectionId: string, _profile: string): Promise<void> => undefined)

vi.mock('@/store/session', () => ({
  $connection,
  $activeSessionId,
  $selectedStoredSessionId,
  setActiveSessionId: (id: null | string) => $activeSessionId.set(id),
  setSelectedStoredSessionId: (id: null | string) => $selectedStoredSessionId.set(id)
}))
vi.mock('@/store/project-scope', () => ({ exitProjectScope: vi.fn() }))
vi.mock('@/store/gateway-switch', () => ({ supersedeGatewaySwitch: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile,
  $freshSessionRequest,
  $newChatProfile: atom(null),
  $newChatRoute: atom(null),
  $showAllProfiles: atom(false),
  captureNewChatSource: vi.fn(),
  currentNewChatIntent: () => 0,
  ensureGatewayAgent,
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  openGatewayAgent,
  refreshActiveProfile: vi.fn(async () => undefined),
  requestFreshSession: vi.fn()
}))

const { _resetConnectionsForTests, initializeConnectionsRegistry } = await import('./connections')
const { _resetFleetRosterForTests } = await import('./fleet-roster')
const { noteUserSessionOpen } = await import('./session-open-intent')

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    { id: 'box', kind: 'ssh', label: 'Box', tokenPreview: null, tokenSet: false }
  ],
  lastUsed: 'box',
  launchMode: 'last-used',
  primary: 'local',
  secureTokenStorage: true,
  version: 2
}

const reachable: DesktopAgentRoster = {
  agents: [],
  sources: [
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
    { connectionId: 'box', kind: 'ssh', label: 'Box', reachable: true, error: 'connect-on-demand' }
  ]
}

const setLastUsed = vi.fn(async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } }))

beforeEach(() => {
  localStorage.clear()
  _resetConnectionsForTests()
  _resetFleetRosterForTests()
  $connection.set({ connectionId: 'local', mode: 'local', registryScoped: true })
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  ensureGatewayAgent.mockClear()
  openGatewayAgent.mockReset()
  setLastUsed.mockClear()
  vi.stubGlobal('window', {
    hermesDesktop: {
      api: vi.fn(async () => ({ profiles: [] })),
      connections: { list: vi.fn(async () => registry), setLastUsed },
      getAgentRoster: async () => reachable
    },
    localStorage,
    location: window.location
  })
})

afterEach(() => vi.unstubAllGlobals())

async function bootWhile(select: () => void): Promise<void> {
  const dial = deferred<void>()
  openGatewayAgent.mockReturnValueOnce(dial.promise)

  const boot = initializeConnectionsRegistry()
  await vi.waitFor(() => expect(openGatewayAgent).toHaveBeenCalledWith('box', 'default'))
  select()
  dial.resolve()
  await boot
}

describe('launch restore onto the last-used gateway', () => {
  it('never closes a chat the person opened while it dialed', async () => {
    await bootWhile(() => {
      // openSession marks the intent at the click; the resume binds later.
      noteUserSessionOpen()
      $selectedStoredSessionId.set('opened-by-user')
    })

    expect($selectedStoredSessionId.get()).toBe('opened-by-user')
    expect($connection.get()?.connectionId).toBe('local')
    expect(setLastUsed).not.toHaveBeenCalled()
  })

  it('still lands over a chat selected automatically, as the cold-start route resume does', async () => {
    await bootWhile(() => $selectedStoredSessionId.set('resumed-from-route'))

    expect($connection.get()?.connectionId).toBe('box')
    expect($selectedStoredSessionId.get()).toBeNull()
    expect(setLastUsed).toHaveBeenCalledWith('box')
  })
})
