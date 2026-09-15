// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from '@/api/client'

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
    connection: makeAtom<null | { baseUrl: string; mode: 'local' | 'remote'; connectionId?: string }>(null),
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
  $connection: mocks.connection,
  $gatewayState: mocks.gatewayState
}))

const { shouldNotify, startMcpHealthChecker, stopMcpHealthChecker } = await import('./mcp-health')

type Status = 'error' | 'needs-auth' | 'ok'

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  setApiRequestLocalMode(true)
  setApiRequestProfile('default')
  mocks.connection.set({ baseUrl: 'http://local.test', mode: 'local' })
})

afterEach(() => {
  stopMcpHealthChecker()
  setApiRequestConnection(null)
  setApiRequestLocalMode(false)
  setApiRequestProfile(null)
  mocks.gatewayState.set('closed')
  mocks.connection.set(null)
  mocks.activeProfile.set('default')
  mocks.getHermesConfigRecord.mockReset()
  mocks.notify.mockReset()
  mocks.testMcpServer.mockReset()
  mocks.setMcpServerEnabled.mockClear()
})

describe('shouldNotify', () => {
  const DAY = 24 * 60 * 60 * 1000
  const now = 1_000_000

  // A future cooldown survives a module reload and suppresses continuing
  // failures, including error <-> needs-auth transitions. Recovery is the
  // exception: a post-recovery incident is immediate.
  it.each<[previous: Status | null, next: Status, notify: boolean]>([
    [null, 'ok', false],
    [null, 'needs-auth', false],
    [null, 'error', false],
    ['ok', 'ok', false],
    ['ok', 'needs-auth', true],
    ['ok', 'error', true],
    ['needs-auth', 'needs-auth', false],
    ['error', 'error', false],
    ['needs-auth', 'error', false],
    ['error', 'needs-auth', false],
    ['needs-auth', 'ok', false],
    ['error', 'ok', false]
  ])('snoozed: previous=%s next=%s → notify=%s', (previous, next, expected) => {
    expect(shouldNotify(previous, next, now + DAY, now)).toBe(expected)
  })

  it('notifies a first failure without a persisted cooldown', () => {
    expect(shouldNotify(null, 'needs-auth', 0, now)).toBe(true)
    expect(shouldNotify(null, 'error', 0, now)).toBe(true)
  })

  it('re-nudges after the daily snooze lapses and starts a new incident after recovery', () => {
    expect(shouldNotify('needs-auth', 'needs-auth', now - 1, now)).toBe(true)
    expect(shouldNotify('error', 'error', now, now)).toBe(true)
    expect(shouldNotify('ok', 'ok', now - DAY, now)).toBe(false)
    expect(shouldNotify('needs-auth', 'ok', 0, now)).toBe(false)
    expect(shouldNotify('ok', 'needs-auth', now + DAY, now)).toBe(true)
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
    expect(mocks.setMcpServerEnabled).toHaveBeenCalledWith('linear', false, {
      profile: 'default'
    })
  } finally {
    nowSpy.mockRestore()
  }
})

it('keeps config, probe, and toast actions pinned to the connection that started the sweep', async () => {
  let releaseConfig!: (config: Record<string, unknown>) => void

  const pendingConfig = new Promise<Record<string, unknown>>(resolve => {
    releaseConfig = resolve
  })

  setApiRequestConnection('connection-a')
  setApiRequestProfile('default')
  mocks.getHermesConfigRecord.mockReturnValueOnce(pendingConfig)
  mocks.testMcpServer.mockResolvedValue({ ok: false, error: 'OAuth: authorization required', tools: [] })

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledWith({ connectionId: 'connection-a', profile: 'default' })

  // The ambient gateway moves while the config read is in flight. The
  // original sweep must still probe and disable on connection A.
  setApiRequestConnection('connection-b')
  releaseConfig({ mcp_servers: { linear: { url: 'https://mcp.linear.app/mcp', auth: 'oauth' } } })
  await flush()
  await flush()

  expect(mocks.testMcpServer).toHaveBeenCalledWith('linear', { connectionId: 'connection-a', profile: 'default' })
  expect(mocks.notify).toHaveBeenCalledTimes(1)
  const toast = mocks.notify.mock.calls[0][0]
  window.location.hash = ''
  toast.action.onClick()
  expect(window.location.hash).toBe('')
  toast.secondaryAction.onClick()
  await flush()
  expect(mocks.setMcpServerEnabled).toHaveBeenCalledWith('linear', false, {
    connectionId: 'connection-a',
    profile: 'default'
  })
})

it('keeps a legacy remote sweep profile-scoped, then drops stale results after its descriptor moves', async () => {
  let releaseConfig!: (config: Record<string, unknown>) => void

  const pendingConfig = new Promise<Record<string, unknown>>(resolve => {
    releaseConfig = resolve
  })

  setApiRequestConnection(null)
  setApiRequestLocalMode(false)
  mocks.connection.set({ baseUrl: 'https://legacy-a.example', mode: 'remote' })
  mocks.getHermesConfigRecord.mockReturnValueOnce(pendingConfig)
  mocks.testMcpServer.mockResolvedValue({ ok: false, error: 'OAuth: authorization required', tools: [] })

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  expect(mocks.getHermesConfigRecord).toHaveBeenCalledWith({ profile: 'default' })

  mocks.connection.set({ baseUrl: 'https://legacy-b.example', mode: 'remote' })
  releaseConfig({ mcp_servers: { legacy: { url: 'https://mcp.example.test/legacy', auth: 'oauth' } } })
  await flush()
  await flush()

  expect(mocks.testMcpServer).not.toHaveBeenCalled()
  expect(mocks.notify).not.toHaveBeenCalled()
})

it('keeps same-profile health status, snooze, and probe entries separate per connection', async () => {
  const servers = { mcp_servers: { shared: { url: 'https://mcp.example.test/mcp' } } }
  mocks.getHermesConfigRecord.mockResolvedValue(servers)
  mocks.testMcpServer.mockResolvedValue({ ok: false, error: 'connection refused', tools: [] })
  setApiRequestProfile('default')

  setApiRequestConnection('connection-a')
  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  await flush()

  setApiRequestConnection('connection-b')
  mocks.gatewayState.set('closed')
  mocks.gatewayState.set('open')
  await flush()
  await flush()

  expect(mocks.testMcpServer).toHaveBeenCalledTimes(2)
  expect(mocks.testMcpServer).toHaveBeenNthCalledWith(1, 'shared', {
    connectionId: 'connection-a',
    profile: 'default'
  })
  expect(mocks.testMcpServer).toHaveBeenNthCalledWith(2, 'shared', {
    connectionId: 'connection-b',
    profile: 'default'
  })
  expect(mocks.notify).toHaveBeenCalledTimes(2)
})

it('keeps legacy local health checks ambient without inventing a local registry pin', async () => {
  mocks.getHermesConfigRecord.mockResolvedValue({
    mcp_servers: { local: { url: 'https://mcp.example.test/local' } }
  })
  mocks.testMcpServer.mockResolvedValue({ ok: true, tools: [] })
  setApiRequestConnection(null)
  setApiRequestLocalMode(true)
  setApiRequestProfile('default')

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  await flush()

  expect(mocks.getHermesConfigRecord).toHaveBeenCalledWith({ profile: 'default' })
  expect(mocks.testMcpServer).toHaveBeenCalledWith('local', { profile: 'default' })
})

it('fails closed when the active owner has no connection identity', async () => {
  setApiRequestConnection(null)
  setApiRequestLocalMode(false)
  mocks.connection.set(null)
  mocks.getHermesConfigRecord.mockResolvedValue({
    mcp_servers: { legacy: { url: 'https://mcp.example.test/legacy' } }
  })

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()

  expect(mocks.getHermesConfigRecord).not.toHaveBeenCalled()
  expect(mocks.testMcpServer).not.toHaveBeenCalled()
  expect(mocks.notify).not.toHaveBeenCalled()
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

it('retains persisted snooze across a real module reload, clears it on recovery, and nudges the next incident', async () => {
  const now = 1_700_000_000_000
  const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(now)
  const servers = { mcp_servers: { reload: { url: 'https://mcp.example.test/reload', auth: 'oauth' } } }
  const failure = { ok: false, error: 'OAuth: authorization required', tools: [] }

  window.localStorage.clear()
  mocks.getHermesConfigRecord.mockResolvedValue(servers)
  mocks.testMcpServer.mockResolvedValue(failure)

  startMcpHealthChecker()
  mocks.gatewayState.set('open')
  await flush()
  await flush()
  expect(mocks.notify).toHaveBeenCalledTimes(1)
  const snoozeKey = 'hermes:mcp-health-snooze-until:local::default::reload'
  expect(window.localStorage.getItem(snoozeKey)).toBe(String(now + 24 * 60 * 60 * 1000))

  stopMcpHealthChecker()
  vi.resetModules()
  let reloadedHealth: { startMcpHealthChecker: () => void; stopMcpHealthChecker: () => void } | null = null

  try {
    const reloadedClient = await import('@/api/client')
    reloadedHealth = await import('./mcp-health')
    const reloadedProbeCache = await import('@/lib/mcp-probe-cache')

    reloadedClient.setApiRequestConnection(null)
    reloadedClient.setApiRequestLocalMode(true)
    reloadedClient.setApiRequestProfile('default')
    mocks.notify.mockReset()
    mocks.testMcpServer.mockReset()
    mocks.testMcpServer.mockResolvedValue(failure)
    mocks.gatewayState.set('closed')
    reloadedHealth.startMcpHealthChecker()
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).not.toHaveBeenCalled()

    reloadedProbeCache.probeCache.clear()
    mocks.testMcpServer.mockResolvedValue({ ok: true, tools: [] })
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(snoozeKey)).toBeNull()

    reloadedProbeCache.probeCache.clear()
    mocks.testMcpServer.mockResolvedValue(failure)
    mocks.gatewayState.set('closed')
    mocks.gatewayState.set('open')
    await flush()
    await flush()
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  } finally {
    reloadedHealth?.stopMcpHealthChecker()
    nowSpy.mockRestore()
  }
})
