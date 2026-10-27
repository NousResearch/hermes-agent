import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// Keep registry activation, descriptor publication and filesystem dispatch real.
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn(),
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => { this.connectionState = 'open' })
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const gateway = await import('./gateway')
const { ensureGatewayAgent, $activeGatewayProfile, $gatewaySwapTarget } = await import('./profile')
const { $connection, setConnection } = await import('./session')
const { desktopFsProfile, isDesktopFsRemoteMode, readDesktopDir } = await import('@/lib/desktop-fs')
const getConnectionFor = vi.fn()

const local = {
  connectionId: 'local', profile: 'default', mode: 'local', baseUrl: '', registryScoped: true
} as HermesConnection

const remote = {
  connectionId: 'old-remote', profile: 'research', mode: 'remote',
  baseUrl: 'https://old.invalid', wsUrl: 'wss://old.invalid/api/ws', registryScoped: true
} as HermesConnection

beforeEach(() => {
  getConnectionFor.mockReset()
  gateway.configureGatewayRegistry({
    onEvent: vi.fn(),
    onActiveRouteChanged: (profile: string) => $activeGatewayProfile.set(profile)
  })
  gateway.setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  gateway.setPrimaryGatewayConnection(local)
  setConnection(remote)
  $activeGatewayProfile.set('research')
  window.hermesDesktop = {
    getConnectionFor,
    api: vi.fn(async () => ({ entries: [] })),
    readDir: vi.fn(async () => ({ entries: [] }))
  } as never
})

afterEach(() => {
  gateway.closeSecondaryGateways()
  $connection.set(null)
  vi.clearAllMocks()
})

it('retries a failed primary descriptor lookup when the same exact route is ensured again', async () => {
  getConnectionFor.mockResolvedValue(remote)
  await ensureGatewayAgent('old-remote', 'research')
  expect(gateway.activeGatewayConnectionId()).toBe('old-remote')
  expect($connection.get()?.connectionId).toBe('old-remote')

  getConnectionFor.mockReset()
  getConnectionFor.mockRejectedValueOnce(new Error('transient descriptor failure')).mockResolvedValue(local)
  await ensureGatewayAgent('local', 'default')
  expect(gateway.activeGatewayConnectionId()).toBe('local')
  expect(gateway.activeGatewayProfileKey()).toBe('default')
  expect($activeGatewayProfile.get()).toBe('default')
  expect(gateway.$gateway.get()?.connectionState).toBe('open')
  // The existing fail-open policy keeps the outgoing descriptor until retry.
  expect($connection.get()?.connectionId).toBe('old-remote')

  await ensureGatewayAgent('local', 'default')
  await readDesktopDir('/descriptor-recovery')
  expect($connection.get()?.connectionId).toBe('local')
  expect(getConnectionFor).toHaveBeenCalledTimes(2)
  expect(desktopFsProfile()).toBe('default')
  expect(isDesktopFsRemoteMode()).toBe(false)
  expect(window.hermesDesktop.readDir).toHaveBeenCalledWith('/descriptor-recovery')
  expect(window.hermesDesktop.api).not.toHaveBeenCalled()
})

it('retries a missing descriptor after fail-open activation', async () => {
  setConnection(null)
  getConnectionFor.mockRejectedValueOnce(new Error('transient descriptor failure')).mockResolvedValue(local)
  await ensureGatewayAgent('local', 'default')
  expect($connection.get()).toBeNull()
  await ensureGatewayAgent('local', 'default')
  expect($connection.get()?.connectionId).toBe('local')
  expect(getConnectionFor).toHaveBeenCalledTimes(2)
})

it('repairs a same-connection descriptor with the wrong profile on an exact route', async () => {
  setConnection({ ...local, profile: 'research' })
  $activeGatewayProfile.set('default')
  getConnectionFor.mockResolvedValue(local)
  await ensureGatewayAgent('local', 'default')
  expect(getConnectionFor).toHaveBeenCalledOnce()
  expect(desktopFsProfile()).toBe('default')
})

it('does not wake an already-ready exact route with a matching descriptor', async () => {
  setConnection(local)
  $activeGatewayProfile.set('default')
  const events: (null | string)[] = []
  const off = $gatewaySwapTarget.listen(value => events.push(value))
  const hook = vi.fn(() => true)

  try {
    await ensureGatewayAgent('local', 'default', { beforeActivate: hook })
  } finally {
    off()
  }

  expect(hook).toHaveBeenCalledOnce()
  expect(getConnectionFor).not.toHaveBeenCalled()
  expect(events).toEqual([])
})

it('does not invoke a ready-route hook for an already-aborted owner', async () => {
  setConnection(local)
  $activeGatewayProfile.set('default')
  const controller = new AbortController()
  controller.abort()
  const hook = vi.fn(() => true)
  await ensureGatewayAgent('local', 'default', { signal: controller.signal, beforeActivate: hook })
  expect(hook).not.toHaveBeenCalled()
  expect(getConnectionFor).not.toHaveBeenCalled()
})

it('a ready-route veto invokes the hook without publication or wake', async () => {
  setConnection(local)
  $activeGatewayProfile.set('default')
  const hook = vi.fn(() => false)
  const off = $connection.listen(() => { throw new Error('unexpected publication') })

  try {
    await ensureGatewayAgent('local', 'default', { beforeActivate: hook })
  } finally {
    off()
  }

  expect(hook).toHaveBeenCalledOnce()
  expect(getConnectionFor).not.toHaveBeenCalled()
})

it('a same-route waiter waits for descriptor publication before evaluating its no-op', async () => {
  let release!: (value: HermesConnection) => void
  getConnectionFor.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  const first = ensureGatewayAgent('local', 'default')
  const hook = vi.fn(() => true)
  const second = ensureGatewayAgent('local', 'default', { beforeActivate: hook })
  await Promise.resolve()
  expect(hook).not.toHaveBeenCalled()
  release(local)
  await first
  await second
  expect(hook).toHaveBeenCalledOnce()
  expect(getConnectionFor).toHaveBeenCalledOnce()
  expect($connection.get()?.connectionId).toBe('local')
})
