import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// pluginSocket must dial the ACTIVE gateway's backend — resolved through the
// same (connectionId, profile) source of truth ensureGatewayProfile /
// ensureGatewayAgent maintain for $connection — not the unscoped primary
// (#73044). Exercises the REAL hermes + store/gateway + store/profile chain:
//  1. A profile switch routes the plugin socket to the pooled profile backend
//     (getConnection(profile), like pluginRest's profileScoped()).
//  2. A registry-agent activation routes it to the agent's SOURCE connection
//     (getConnectionFor), not the local pool — the post-#87600 shape.

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    // Stub only the socket class so gateway activations don't dial real WS.
    HermesGateway: class {
      connectionState = 'closed'
      connect = async (_wsUrl: string): Promise<void> => {
        this.connectionState = 'open'
      }
      close = (): void => {
        this.connectionState = 'closed'
      }
      onEvent = vi.fn(() => () => {})
      onState = vi.fn(() => () => {})
    }
  }
})
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { pluginSocket, setApiRequestConnection, setApiRequestProfile } = await import('@/hermes')
const { closeSecondaryGateways, configureGatewayRegistry, setPrimaryGateway } = await import('@/store/gateway')
const { $activeGatewayProfile, ensureGatewayAgent, ensureGatewayProfile } = await import('@/store/profile')

// authMode 'oauth' makes pluginSocket stop after resolving the connection
// (polling fallback), so the assertions cover resolution without a WS dial.
const conn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({
    authMode: 'oauth',
    baseUrl: 'https://pool.invalid',
    mode: 'remote',
    token: 'fake-test-token',
    wsUrl: 'wss://pool.invalid/api/ws?token=fake-test-token',
    ...over
  }) as HermesConnection

let getConnection: ReturnType<typeof vi.fn>
let getConnectionFor: ReturnType<typeof vi.fn>

beforeEach(() => {
  getConnection = vi.fn(async (profile?: null | string) => conn({ profile: profile ?? 'default' }))
  getConnectionFor = vi.fn(async ({ profile }: { connectionId?: null | string; profile?: null | string }) =>
    conn({ baseUrl: 'https://homelab.invalid', profile: profile ?? 'default' })
  )
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnection,
      getConnectionFor,
      getGatewayWsUrl: vi.fn(async () => 'wss://pool.invalid/api/ws?ticket=fake'),
      getGatewayWsUrlFor: vi.fn(async () => 'wss://homelab.invalid/api/ws?ticket=fake'),
      touchBackend: vi.fn(async () => ({ ok: true }))
    }
  })
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  setApiRequestProfile(null)
  setApiRequestConnection(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.restoreAllMocks()
})

describe('pluginSocket active-backend scoping (#73044)', () => {
  it('dials the active profile backend after a profile switch', async () => {
    await ensureGatewayProfile('work')
    getConnection.mockClear()
    getConnectionFor.mockClear()

    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalled())
    expect(getConnection).toHaveBeenCalledWith('work')
    expect(getConnectionFor).not.toHaveBeenCalled()

    dispose()
  })

  it("dials the agent's SOURCE connection after a registry-agent activation", async () => {
    await ensureGatewayAgent('homelab', 'research')
    getConnection.mockClear()
    getConnectionFor.mockClear()

    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(getConnectionFor).toHaveBeenCalled())
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'research' })
    expect(getConnection).not.toHaveBeenCalled()

    dispose()
  })

  it('falls back to the primary when no profile or connection is active', async () => {
    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalled())
    expect(getConnection).toHaveBeenCalledWith(null)

    dispose()
  })
})
