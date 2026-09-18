// @vitest-environment jsdom
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
    notify: vi.fn(),
    setMcpServerEnabled: vi.fn().mockResolvedValue({ ok: true }),
    testMcpServer: vi.fn()
  }
})

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: mocks.getHermesConfigRecord,
  setMcpServerEnabled: mocks.setMcpServerEnabled,
  testMcpServer: mocks.testMcpServer
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: vi.fn()
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: mocks.activeProfile,
  normalizeProfileKey: (name: string | null | undefined) => (name ?? '').trim() || 'default'
}))

vi.mock('@/store/session', () => ({
  $gatewayState: mocks.gatewayState
}))

const { shouldNotify, startMcpHealthChecker, stopMcpHealthChecker } = await import('./mcp-health')

type Status = 'error' | 'needs-auth' | 'ok'

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

afterEach(() => {
  stopMcpHealthChecker()
  mocks.gatewayState.set('closed')
  mocks.activeProfile.set('default')
  mocks.getHermesConfigRecord.mockReset()
  mocks.notify.mockReset()
  mocks.testMcpServer.mockReset()
  mocks.setMcpServerEnabled.mockClear()
})

type McpHealthModule = Awaited<ReturnType<typeof importMcpHealth>>
const importMcpHealth = () => import('./mcp-health')

describe('shouldNotify', () => {
  const DAY = 24 * 60 * 60 * 1000
  const now = 1_000_000

  // With an active snooze, a fresh session and unchanged bad state stay quiet;
  // later status transitions still notify. Ok never nudges.
  it.each<[previous: Status | null, next: Status, notify: boolean]>([
    [null, 'ok', false],
    [null, 'needs-auth', false],
    [null, 'error', false],
    ['ok', 'ok', false],
    ['ok', 'needs-auth', true],
    ['ok', 'error', true],
    ['needs-auth', 'needs-auth', false],
    ['error', 'error', false],
    ['needs-auth', 'error', true],
    ['error', 'needs-auth', true],
    ['needs-auth', 'ok', false],
    ['error', 'ok', false]
  ])('snoozed: previous=%s next=%s → notify=%s', (previous, next, expected) => {
    expect(shouldNotify(previous, next, now + DAY, now)).toBe(expected)
  })

  it('notifies a newly discovered bad server when no snooze has been persisted', () => {
    expect(shouldNotify(null, 'needs-auth', 0, now)).toBe(true)
    expect(shouldNotify(null, 'error', 0, now)).toBe(true)
  })

  it('re-nudges a server that stays broken once the daily snooze lapses, never for ok', () => {
    expect(shouldNotify('needs-auth', 'needs-auth', now - 1, now)).toBe(true)
    expect(shouldNotify('error', 'error', now, now)).toBe(true)
    expect(shouldNotify('ok', 'ok', now - DAY, now)).toBe(false)
    expect(shouldNotify('needs-auth', 'ok', 0, now)).toBe(false)
  })
})

it('shows the toast with Sign in + Disable, then stays quiet for a day and re-nudges after it', async () => {
  const servers = { mcp_servers: { linear: { url: 'https://mcp.linear.app/mcp', auth: 'oauth' } } }
  mocks.getHermesConfigRecord.mockResolvedValue(servers)
  mocks.testMcpServer.mockResolvedValue({ ok: false, error: 'OAuth: authorization required', tools: [] })
  window.localStorage.clear()

  let clock = 1_700_000_000_000
  const nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => clock)

  try {
    startMcpHealthChecker()
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).toHaveBeenCalledTimes(1)
    const toast = mocks.notify.mock.calls[0][0]
    expect(toast.action.label).toBe('notifications.mcp.signIn')
    expect(toast.secondaryAction.label).toBe('notifications.mcp.disable')

    // Same day, still broken: a reconnect sweep must not re-pop.
    clock += 60 * 60 * 1000
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).toHaveBeenCalledTimes(1)

    // Next day, still broken: one more nudge.
    clock += 24 * 60 * 60 * 1000
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).toHaveBeenCalledTimes(2)

    // Disable from the toast flips enabled:false on the backend.
    toast.secondaryAction.onClick()
    await flush()
    expect(mocks.setMcpServerEnabled).toHaveBeenCalledWith('linear', false)
  } finally {
    nowSpy.mockRestore()
  }
})

it('honors a persisted snooze in a fresh module session, then re-notifies after it expires', async () => {
  const servers = { mcp_servers: { linear: { url: 'https://mcp.linear.app/mcp', auth: 'oauth' } } }
  const key = 'hermes:mcp-health-snooze-until:default::linear'
  let clock = 1_700_000_000_000
  const until = clock + 24 * 60 * 60 * 1000
  const nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => clock)
  let freshSession: McpHealthModule | undefined

  try {
    window.localStorage.setItem(key, String(until))
    mocks.getHermesConfigRecord.mockResolvedValue(servers)
    mocks.testMcpServer.mockResolvedValue({ ok: false, error: 'OAuth: authorization required', tools: [] })

    // A fresh import drops in-memory transition state, as a renderer restart does.
    vi.resetModules()
    freshSession = await importMcpHealth()
    freshSession.startMcpHealthChecker()
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).not.toHaveBeenCalled()

    clock = until + 1
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  } finally {
    freshSession?.stopMcpHealthChecker()
    nowSpy.mockRestore()
  }
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

it('runs one follow-up when the active sweep fails through the handled config-error path', async () => {
  let rejectFirst!: (err: Error) => void

  const first = new Promise<Record<string, unknown>>((_resolve, reject) => {
    rejectFirst = reject
  })

  mocks.getHermesConfigRecord.mockReturnValueOnce(first).mockResolvedValue({ mcp_servers: {} })

  startMcpHealthChecker()
  mocks.gatewayState.set('open')

  for (let index = 0; index < 12; index += 1) {
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
  }

  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(1)

  rejectFirst(new Error('backend restarting'))
  await flush()
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledTimes(2)
})
