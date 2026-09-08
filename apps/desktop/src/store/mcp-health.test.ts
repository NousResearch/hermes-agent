import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => {
  const makeAtom = <T>(initial: T) => {
    let value = initial
    const listeners = new Set<(value: T) => void>()

    return {
      get: () => value,
      listen(listener: (value: T) => void) {
        listeners.add(listener)

        return () => listeners.delete(listener)
      },
      set(next: T) {
        value = next

        for (const listener of listeners) {
          listener(value)
        }
      },
      subscribe(listener: (value: T) => void) {
        listener(value)

        return this.listen(listener)
      }
    }
  }

  return {
    activeProfile: makeAtom('default'),
    gatewayState: makeAtom<'closed' | 'open'>('closed'),
    getHermesConfigRecord: vi.fn(),
    notify: vi.fn()
  }
})

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: mocks.getHermesConfigRecord,
  testMcpServer: vi.fn()
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: mocks.activeProfile,
  normalizeProfileKey: (name: string | null | undefined) => (name ?? '').trim() || 'default'
}))

vi.mock('@/store/session', () => ({
  $gatewayState: mocks.gatewayState
}))

const { shouldNotifyOnTransition, startMcpHealthChecker, stopMcpHealthChecker } = await import('./mcp-health')

type Status = 'error' | 'needs-auth' | 'ok'

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

afterEach(() => {
  stopMcpHealthChecker()
  mocks.gatewayState.set('closed')
  mocks.activeProfile.set('default')
  mocks.getHermesConfigRecord.mockReset()
  mocks.notify.mockReset()
})

describe('shouldNotifyOnTransition', () => {
  it.each<[previous: Status | null, next: Status, notify: boolean]>([
    [null, 'ok', false],
    [null, 'needs-auth', true],
    [null, 'error', true],
    ['ok', 'ok', false],
    ['ok', 'needs-auth', true],
    ['ok', 'error', true],
    ['needs-auth', 'needs-auth', false],
    ['error', 'error', false],
    ['needs-auth', 'error', true],
    ['error', 'needs-auth', true],
    ['needs-auth', 'ok', false],
    ['error', 'ok', false]
  ])('previous=%s next=%s → notify=%s', (previous, next, expected) => {
    expect(shouldNotifyOnTransition(previous, next)).toBe(expected)
  })
})

it('coalesces reconnects during a sweep into one fresh follow-up sweep', async () => {
  let releaseFirst!: (config: Record<string, unknown>) => void

  const first = new Promise<Record<string, unknown>>(resolve => {
    releaseFirst = resolve
  })

  mocks.getHermesConfigRecord.mockReturnValueOnce(first).mockResolvedValue({ mcp_servers: {} })

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(1)

  for (let index = 0; index < 12; index += 1) {
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
  }

  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(1)

  releaseFirst({ mcp_servers: {} })
  await flush()
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(2)
})

it.each(['stop', 'disconnect'])('drops the queued sweep on %s', async action => {
  let release!: (config: Record<string, unknown>) => void
  mocks.getHermesConfigRecord.mockReturnValueOnce(new Promise(resolve => { release = resolve })).mockResolvedValue({ mcp_servers: {} })
  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  mocks.gatewayState.set('closed')
  mocks.gatewayState.set('open')

  if (action === 'stop') {stopMcpHealthChecker()}
  else {mocks.gatewayState.set('closed')}

  release({ mcp_servers: {} })
  await flush()
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(1)
})
