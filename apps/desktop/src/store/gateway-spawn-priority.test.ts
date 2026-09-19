import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// #102281: a user-initiated open must reach Electron main as a FOREGROUND dial
// on its FIRST IPC, not only on the secondary's connect. Every open first
// probes the route (sharedPrimaryRoute / isAttachedSharedRemote) with
// getConnection / getConnectionFor; if that probe is untagged, main starts the
// spawn as a background slot wait and the click waits out the probe's 20s
// timeout before anything promotes it.

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async (): Promise<void> => {
      this.connectionState = 'open'
    }
    close = (): void => {
      this.connectionState = 'closed'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
    request = vi.fn(async () => ({}))
  }
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForAgent,
  openGatewayForProfile,
  requestGatewayForAgent,
  retainGatewayForAgent,
  setPrimaryGateway
} = await import('./gateway')

const conn = {
  authMode: 'token',
  baseUrl: 'https://homelab.invalid',
  mode: 'remote',
  profile: 'research',
  token: 'fake-test-token',
  wsUrl: 'wss://homelab.invalid/api/ws?token=fake-test-token'
}

function installDesktop(): { getConnection: ReturnType<typeof vi.fn>; getConnectionFor: ReturnType<typeof vi.fn> } {
  const stub = {
    getConnection: vi.fn(async () => conn),
    getConnectionFor: vi.fn(async () => conn)
  }

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub

  return stub
}

function priorities(mock: ReturnType<typeof vi.fn>, pick: (args: unknown[]) => unknown): unknown[] {
  return mock.mock.calls.map(args => pick(args))
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('user opens dial main as foreground from the first IPC (#102281)', () => {
  it('ensureGatewayForProfile tags the route probe AND the connect dial', async () => {
    const desktop = installDesktop()

    await ensureGatewayForProfile('research')

    const seen = priorities(desktop.getConnection, args => (args[1] as { priority?: string } | undefined)?.priority)
    expect(seen.length).toBeGreaterThanOrEqual(2)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })

  it('openGatewayForProfile without a priority never tags a dial as foreground', async () => {
    const desktop = installDesktop()

    await openGatewayForProfile('research')

    const seen = priorities(desktop.getConnection, args => (args[1] as { priority?: string } | undefined)?.priority)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === undefined)).toBe(true)
  })

  it('openGatewayForAgent forwards spawnPriority to every registry dial', async () => {
    const desktop = installDesktop()

    await openGatewayForAgent('homelab', 'research', { spawnPriority: 'foreground' })

    const seen = priorities(desktop.getConnectionFor, args => (args[0] as { priority?: string }).priority)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })

  it('ensureGatewayForAgent is always a foreground open', async () => {
    const desktop = installDesktop()

    await ensureGatewayForAgent('homelab', 'research')

    const seen = priorities(desktop.getConnectionFor, args => (args[0] as { priority?: string }).priority)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })
})

// requestGatewayForAgent / retainGatewayForAgent are the RPC-lease pair that
// createBackendSessionForSend, openNewSessionTile and the plugin SDK's
// host.requestProfile dial through. They are not activation doors, so they
// never hardcode 'foreground'; a user gesture (first send, "New session", a
// Bot Chat roster click) passes { spawnPriority: 'foreground' } and every dial
// the call makes — the shared-remote probe and the secondary connect — must
// carry it. Untagged callers keep main's pre-priority IPC payload (#105104).
describe('RPC-lease dials forward an explicit spawnPriority (#105104)', () => {
  const registryPriorities = (desktop: ReturnType<typeof installDesktop>) =>
    priorities(desktop.getConnectionFor, args => (args[0] as { priority?: string }).priority)

  it('requestGatewayForAgent tags the probe and the connect when the caller says foreground', async () => {
    const desktop = installDesktop()

    await requestGatewayForAgent('homelab', 'research', 'session.create', {}, undefined, undefined, {
      spawnPriority: 'foreground'
    })

    const seen = registryPriorities(desktop)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })

  it('requestGatewayForAgent without options never tags a registry dial', async () => {
    const desktop = installDesktop()

    await requestGatewayForAgent('homelab', 'research', 'session.create', {})

    const seen = registryPriorities(desktop)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === undefined)).toBe(true)
  })

  it('retainGatewayForAgent tags the lease dial when the caller says foreground', async () => {
    const desktop = installDesktop()

    const release = await retainGatewayForAgent('homelab', 'research', { spawnPriority: 'foreground' })
    release()

    const seen = registryPriorities(desktop)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })

  it('retainGatewayForAgent without options never tags a registry dial', async () => {
    const desktop = installDesktop()

    const release = await retainGatewayForAgent('homelab', 'research')
    release()

    const seen = registryPriorities(desktop)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === undefined)).toBe(true)
  })

  it('retainGatewayForAgent on the plain-profile route (null connection) still dials foreground', async () => {
    // Neither #105390 nor #110354 covered this branch: the v1 profile resolver
    // goes through gatewayForProfile, not the registry secondary.
    const desktop = installDesktop()

    const release = await retainGatewayForAgent(null, 'research', { spawnPriority: 'foreground' })
    release()

    const seen = priorities(desktop.getConnection, args => (args[1] as { priority?: string } | undefined)?.priority)
    expect(seen.length).toBeGreaterThanOrEqual(1)
    expect(seen.every(priority => priority === 'foreground')).toBe(true)
  })
})
