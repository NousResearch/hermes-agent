import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Profile-door activation failure surfacing (#81094): when a secondary's
// socket cannot be opened, ensureGatewayForProfile must rethrow (after
// arming the reconnect schedule) so the caller can surface the failure,
// and profile.ts must NOT publish an activation for a backend that never
// came up — previously the catch swallowed the error and setActive() ran
// anyway, silently routing messages to the primary socket.

const gatewayMocks = vi.hoisted(() => {
  const instances: { close: ReturnType<typeof vi.fn>; connectionState: string }[] = []

  return {
    connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
    instances
  }
})

vi.mock('@/hermes', () => ({
  setApiRequestProfile: vi.fn(),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  HermesGateway: class {
    connectionState = 'closed'
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
    constructor() {
      gatewayMocks.instances.push(this as never)
    }
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))

const { ensureGatewayProfile } = await import('./profile')

function installDesktop(stub: Record<string, unknown>): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function descriptorFor(profile: string) {
  return {
    authMode: 'token',
    baseUrl: `https://${profile}.invalid`,
    mode: 'local',
    profile,
    token: 'fake-test-token',
    wsUrl: `wss://${profile}.invalid/ws`
  }
}

beforeEach(() => {
  gatewayMocks.instances.length = 0
})

afterEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ensureGatewayProfile — switch failure surfaces instead of silent fallback (#81094)', () => {
  it('rejects when the target backend cannot be dialed, instead of resolving silently', async () => {
    const getConnection = vi.fn(async ({ profile }: { profile: string }) => descriptorFor(profile))
    installDesktop({ getConnection })

    // The secondary dial fails at connect(): the whole switch must reject.
    gatewayMocks.connect.mockRejectedValue(new Error('backend unreachable'))

    await expect(ensureGatewayProfile('work')).rejects.toThrow('backend unreachable')
  })
})
