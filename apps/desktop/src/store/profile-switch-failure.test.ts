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

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
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

// Load after the transport mock is installed; activation must use the real store graph.
const { $activeGatewayProfile, $freshSessionRequest, ensureGatewayProfile, selectProfile } = await import('./profile')
const { activeGatewayProfileKey, closeSecondaryGateways, setPrimaryGateway, setPrimaryGatewayConnectionId } =
  await import('./gateway')
const { $profileSwitchFailure, $profileSwitchPending } = await import('./profile-switch-state')
const { HermesGateway } = await import('@/hermes')

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
  closeSecondaryGateways()
  setPrimaryGateway(null)
  setPrimaryGatewayConnectionId(null)
  vi.resetAllMocks()
  gatewayMocks.connect.mockResolvedValue(undefined)
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

it('offers Retry after a registry rail switch fails to open its socket without replacing the draft', async () => {
  const primary = new HermesGateway()
  await primary.connect('wss://default.invalid/ws')
  setPrimaryGateway(primary)
  setPrimaryGatewayConnectionId('remote-test', 'remote')
  $activeGatewayProfile.set('default')
  installDesktop({ getConnectionFor: async () => ({ ...descriptorFor('research'), connectionId: 'remote-test' }) })
  const dial = Promise.withResolvers<void>()
  gatewayMocks.connect.mockReturnValueOnce(dial.promise)

  selectProfile('research')
  await vi.waitFor(() => expect(gatewayMocks.connect).toHaveBeenCalledWith('wss://research.invalid/ws'))
  expect($profileSwitchPending.get()).toBe('research')
  const drafts = $freshSessionRequest.get()
  dial.reject(new Error('backend unreachable'))
  await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
  expect(activeGatewayProfileKey()).toBe('default')
  const failure = $profileSwitchFailure.get()
  expect(failure?.profile).toBe('research')

  failure?.retry()
  await vi.waitFor(() => expect($activeGatewayProfile.get()).toBe('research'))
  expect(gatewayMocks.connect.mock.calls.filter(([url]) => url === 'wss://research.invalid/ws')).toHaveLength(2)
  expect($profileSwitchPending.get()).toBeNull()
  expect($profileSwitchFailure.get()).toBeNull()
  expect($freshSessionRequest.get()).toBe(drafts)
})
