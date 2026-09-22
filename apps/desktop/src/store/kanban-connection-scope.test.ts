import { QueryObserver } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// A connection switch must leave every profile-scoped query refetched against
// the NEW gateway — see the $activeConnectionId.listen comment in
// store/connections.ts for why the switch's own wipe is not enough.
//
// Real store chain (store/gateway + store/profile + store/connections), only
// the HermesGateway socket class stubbed — same harness as
// plugin-socket-scope.test.ts.

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
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { getApiRequestConnection, setApiRequestConnection, setApiRequestProfile } = await import('@/api/client')
const { queryClient } = await import('@/lib/query-client')
const { closeSecondaryGateways, configureGatewayRegistry, setPrimaryGateway } = await import('@/store/gateway')
const { $activeGatewayProfile } = await import('@/store/profile')
const { selectConnection, setConnectionsRegistry, _resetConnectionsForTests } = await import('@/store/connections')
const { setConnection } = await import('@/store/session')

const conn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({
    authMode: 'oauth',
    baseUrl: 'https://pool.invalid',
    mode: 'remote',
    token: 'fake-test-token',
    wsUrl: 'wss://pool.invalid/api/ws?token=fake-test-token',
    ...over
  }) as HermesConnection

const registry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    { id: 'spark', kind: 'remote', label: 'Spark', tokenPreview: '...abc', tokenSet: true }
  ],
  primary: 'local',
  secureTokenStorage: true,
  version: 2
} as never

describe('connection-switch query invalidation', () => {
  let tagsAtFetch: Array<null | string>
  let observer: QueryObserver<any, any, any, any, any> | undefined

  beforeEach(() => {
    vi.stubGlobal('window', {
      hermesDesktop: {
        api: vi.fn(async () => ({})),
        connections: {
          list: vi.fn(async () => registry),
          setLastUsed: vi.fn(async () => ({ ok: true, registry }))
        },
        getConnection: vi.fn(async (profile?: null | string) =>
          conn({ baseUrl: 'http://127.0.0.1:8117', profile: profile ?? 'default' })
        ),
        getConnectionFor: vi.fn(async ({ connectionId }: { connectionId?: null | string }) =>
          conn({
            baseUrl: connectionId === 'spark' ? 'https://spark.invalid' : 'http://127.0.0.1:8117',
            connectionId: connectionId ?? undefined,
            profile: 'default',
            registryScoped: true
          })
        ),
        getGatewayWsUrl: vi.fn(async () => 'wss://pool.invalid/api/ws?ticket=fake'),
        getGatewayWsUrlFor: vi.fn(async () => 'wss://spark.invalid/api/ws?ticket=fake'),
        touchBackend: vi.fn(async () => ({ ok: true }))
      },
      localStorage: new Map<string, string>() as unknown as Storage
    })
    configureGatewayRegistry({ onEvent: vi.fn() })
    setPrimaryGateway({ connectionState: 'open' } as never, 'default')
    setConnectionsRegistry(registry)
    queryClient.clear()
    tagsAtFetch = []
  })

  afterEach(() => {
    observer?.destroy()
    queryClient.clear()
    closeSecondaryGateways()
    $activeGatewayProfile.set('default')
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    _resetConnectionsForTests()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('refetches connection-scoped queries against the NEW gateway after a switch', async () => {
    // One active profile-scoped query whose connection tag is read at queryFn
    // time, which is what pluginRest → hermesApi does.
    observer = new QueryObserver(queryClient, {
      queryKey: ['kanban', 'boards'],
      queryFn: async () => {
        tagsAtFetch.push(getApiRequestConnection())

        return { boards: [] }
      },
      staleTime: 30_000
    })
    const unsubscribe = observer.subscribe(() => undefined)

    // Let the initial fetch (old backend) settle.
    await vi.waitFor(() => expect(tagsAtFetch).toEqual([null]))
    tagsAtFetch.length = 0

    await selectConnection('spark', { profile: 'default' })

    // The switch's own pre-commit wipe refetches early on the OLD tag; the
    // connection twin must invalidate again once the tag names spark, so at
    // least one fetch rides the new backend and the FINAL fetch is
    // spark-tagged.
    await vi.waitFor(() => expect(tagsAtFetch.at(-1)).toBe('spark'), { timeout: 2_000 })

    unsubscribe()
  })

  it('does not refetch when the connection id has not actually changed', async () => {
    let fetches = 0
    observer = new QueryObserver(queryClient, {
      queryKey: ['kanban', 'boards'],
      queryFn: async () => {
        fetches += 1

        return { boards: [] }
      },
      staleTime: 30_000
    })
    const unsubscribe = observer.subscribe(() => undefined)

    // Settle, not just start: an invalidation racing an in-flight fetch is
    // absorbed by the fetch and would hide a listener that fires too often.
    await vi.waitFor(() => expect(observer!.getCurrentResult().status).toBe('success'))
    expect(fetches).toBe(1)

    // A fresh descriptor object naming the SAME connection id (a resync, not a
    // switch) must not re-invalidate.
    setConnection(connection => ({ ...connection! }))

    await new Promise(resolve => setTimeout(resolve, 200))
    expect(fetches).toBe(1)

    unsubscribe()
  })
})
