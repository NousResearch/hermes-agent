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
