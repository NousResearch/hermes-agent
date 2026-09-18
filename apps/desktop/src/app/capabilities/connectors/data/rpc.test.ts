import { JsonRpcGatewayError } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The page has no chat session, so every operation call must name the ACCOUNT
// owner: a session owner would be refused as NOT_OWNER, and the backend would
// have nowhere to broadcast the operation's frames.
const gatewayMocks = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn(async () => ({}) as unknown)
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

const { asConnectorError, ConnectorRpcError, connectAccountConnectors, isConnectorReason, listConnectors, setConnectorPolicy } =
  await import('./rpc')

const lastCall = () => gatewayMocks.requestGatewayForAgent.mock.calls.at(-1) as unknown as [
  null | string,
  string,
  string,
  Record<string, unknown>
]

describe('connector RPC calls', () => {
  beforeEach(() => {
    gatewayMocks.requestGatewayForAgent.mockClear()
    gatewayMocks.requestGatewayForAgent.mockResolvedValue({})
  })

  it('names the account owner and routes to the scoped profile', async () => {
    await listConnectors('omar')

    const [connectionId, profile, method, params] = lastCall()

    expect([connectionId, profile, method]).toEqual([null, 'omar', 'connectors.list'])
    expect(params).toEqual({ owner: { type: 'account' } })
  })

  it('splits an explicit (connection, profile) pin into the two arguments the router takes', async () => {
    await listConnectors({ connectionId: 'laptop', profile: 'work' })

    expect(lastCall().slice(0, 2)).toEqual(['laptop', 'work'])
  })

  it('carries the account owner into the connect call', async () => {
    await connectAccountConnectors('omar', ['gmail'], true)

    expect(lastCall()[3]).toEqual({ connectors: ['gmail'], owner: { type: 'account' }, reconnect: true })
  })

  it('omits the compare-and-set token rather than sending an empty one', async () => {
    await setConnectorPolicy('omar', { connector: 'gmail', enabled: false, type: 'connector' })

    expect(lastCall()[3]).toEqual({ change: { connector: 'gmail', enabled: false, type: 'connector' } })
  })
})

describe('connector errors', () => {
  it('keeps the backend reason so the page can branch on it', async () => {
    gatewayMocks.requestGatewayForAgent.mockRejectedValueOnce(
      new JsonRpcGatewayError('Connector policy changed.', { code: 4090, data: { reason: 'POLICY_CONFLICT' } })
    )

    const error = await listConnectors('omar').catch((thrown: unknown) => thrown)

    expect(error).toBeInstanceOf(ConnectorRpcError)
    expect(isConnectorReason(error, 'POLICY_CONFLICT')).toBe(true)
    expect(isConnectorReason(error, 'NEEDS_NOUS_AUTH')).toBe(false)
  })

  it('leaves the reason unset for a failure that never reached a connector handler', () => {
    // A timeout, a closed socket, or a reason this build has never heard of: the
    // page must fall through to its generic failure, never to a typed state.
    expect(asConnectorError(new Error('gateway not connected')).reason).toBeUndefined()
    expect(
      asConnectorError(new JsonRpcGatewayError('nope', { data: { reason: 'SOMETHING_NEW' } })).reason
    ).toBeUndefined()
  })
})
