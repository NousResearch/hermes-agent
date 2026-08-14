import { beforeEach, describe, expect, it, vi } from 'vitest'

const instances: FakeGateway[] = []

class FakeGateway {
  connectionState = 'closed'
  close = vi.fn(() => {
    this.connectionState = 'closed'
  })
  connect = vi.fn(async () => {
    this.connectionState = 'open'
  })
  onEvent = vi.fn(() => () => undefined)
  onState = vi.fn(() => () => undefined)

  constructor() {
    instances.push(this)
  }
}

vi.mock('@hermes/shared', () => ({
  resolveGatewayWsUrl: vi.fn(async () => 'ws://local.test/api/ws')
}))
vi.mock('@/hermes', () => ({ HermesGateway: FakeGateway }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/store/session', () => ({ setGatewayState: vi.fn() }))

describe('profile gateway rehome', () => {
  beforeEach(() => {
    instances.length = 0
    vi.resetModules()
  })

  it('discards a stale open racing an exact-profile Apply and keeps the replacement', async () => {
    let resolveFirst!: (value: { baseUrl: string; mode: 'local'; token: string; wsUrl: string }) => void
    const firstConnection = new Promise<{ baseUrl: string; mode: 'local'; token: string; wsUrl: string }>(resolve => {
      resolveFirst = resolve
    })
    const getConnection = vi
      .fn()
      .mockReturnValueOnce(firstConnection)
      .mockResolvedValue({ baseUrl: 'http://local.test', mode: 'local', token: 'new', wsUrl: 'ws://new' })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection, touchBackend: vi.fn(async () => undefined) }
    })

    const { openGatewayForProfile, rehomeSecondaryGateway, setPrimaryGateway } = await import('./gateway')

    setPrimaryGateway(new FakeGateway() as never, 'default')
    const staleOpen = openGatewayForProfile('work')
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(1))

    const rehome = rehomeSecondaryGateway('work')
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))
    resolveFirst({ baseUrl: 'http://stale.test', mode: 'local', token: 'old', wsUrl: 'ws://old' })
    await Promise.all([staleOpen, rehome])

    const stale = instances[1]
    const replacement = instances[2]

    expect(stale.connect).not.toHaveBeenCalled()
    expect(stale.close).toHaveBeenCalledOnce()
    expect(replacement.connect).toHaveBeenCalledOnce()
    expect(replacement.close).not.toHaveBeenCalled()
  })
})
