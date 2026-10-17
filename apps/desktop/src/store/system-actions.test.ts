import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { confirm, getActionStatus, getStatus, notify, notifyError, restartGateway } = vi.hoisted(() => ({
  confirm: vi.fn(),
  getActionStatus: vi.fn(),
  getStatus: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  restartGateway: vi.fn()
}))

vi.mock('@/hermes', () => ({ getActionStatus, getStatus, restartGateway }))
vi.mock('@/store/confirm', () => ({ confirm }))
vi.mock('@/store/notifications', () => ({ notify, notifyError }))

import { registerGatewayReconnect } from '@/store/gateway-reconnect'

import { $gatewayRestarting, runGatewayRestart, watchGatewayRestartOutcome } from './system-actions'

const POLL_WINDOW_MS = 18 * 1_200
const settlePollWindow = () => vi.advanceTimersByTimeAsync(POLL_WINDOW_MS + 1_000)

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}

const backendDownUntil = (healthyAt: number) => {
  let polls = 0

  getActionStatus.mockImplementation(async () => {
    polls += 1

    if (polls < healthyAt) {throw new Error('fetch failed')}

    return { name: 'gateway-restart', running: true, exit_code: null, pid: 4242, lines: [] }
  })
}

const refusedBackend = () => getActionStatus.mockRejectedValue(new Error('fetch failed'))

const freshProcess = () =>
  getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: null, pid: null, lines: [] })

afterEach(() => {
  vi.useRealTimers()
  vi.resetAllMocks()
})

describe('runGatewayRestart during the restart window (#123111)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    getStatus.mockResolvedValue({})
    restartGateway.mockResolvedValue({ ok: true, pid: 4242, name: 'gateway-restart' })
  })

  it('resolves success when the status poll is refused mid-window, then answers', async () => {
    backendDownUntil(4)
    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(true)
    expect(getActionStatus).toHaveBeenCalled()
  })

  it('resolves success when a fresh process reports the action unknown', async () => {
    freshProcess()
    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(true)
  })

  it('fails when the poll window is refused end to end', async () => {
    refusedBackend()
    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('still surfaces a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })
    const outcome = runGatewayRestart()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('hands reconnection to the gateway reconnect owner after a confirmed restart', async () => {
    backendDownUntil(4)
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayRestart()
      await settlePollWindow()
      await expect(outcome).resolves.toBe(true)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })

  it('hands reconnection to the owner even when the recorded exit failed', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayRestart()
      await settlePollWindow()
      await expect(outcome).resolves.toBe(false)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })
})

describe('watchGatewayRestartOutcome', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  it('resolves true across a refused-then-answered poll window and reconnects', async () => {
    backendDownUntil(4)
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = watchGatewayRestartOutcome()
      await settlePollWindow()
      await expect(outcome).resolves.toBe(true)
      expect(handler).toHaveBeenCalledOnce()
    } finally {
      off()
    }
  })

  it('resolves false when the poll window is refused end to end', async () => {
    refusedBackend()
    const outcome = watchGatewayRestartOutcome()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })

  it('resolves false on a recorded non-zero exit', async () => {
    getActionStatus.mockResolvedValue({ name: 'gateway-restart', running: false, exit_code: 1, pid: null, lines: [] })
    const outcome = watchGatewayRestartOutcome()

    await settlePollWindow()
    await expect(outcome).resolves.toBe(false)
  })
})

describe('runGatewayRestart owner lifetime', () => {
  beforeEach(() => {
    getStatus.mockResolvedValue({ gateway_shared_with: ['default', 'research'] })
    getActionStatus.mockResolvedValue({ running: false, exit_code: 0 })
    restartGateway.mockResolvedValue({ name: 'gateway-restart' })
  })

  it('does not restart after its owner changes while confirmation is open', async () => {
    const approval = deferred<boolean>()
    let current = true
    confirm.mockReturnValue(approval.promise)

    const pending = runGatewayRestart({ connectionId: 'owner-a', profile: 'default' }, () => current)
    await vi.waitFor(() => expect(confirm).toHaveBeenCalled())
    current = false
    approval.resolve(true)

    await expect(pending).resolves.toBe(false)
    expect(restartGateway).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('suppresses a restart rejection after its owner changes', async () => {
    const restart = deferred<never>()
    let current = true
    getStatus.mockResolvedValue({ gateway_shared_with: null })
    restartGateway.mockReturnValue(restart.promise)

    const pending = runGatewayRestart({ connectionId: 'owner-a', profile: 'default' }, () => current)
    await vi.waitFor(() => expect(restartGateway).toHaveBeenCalled())
    current = false
    restart.reject(new Error('owner A stopped'))

    await expect(pending).resolves.toBe(false)
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('keeps restart progress visible while another restart is still pending', async () => {
    const first = deferred<never>()
    const second = deferred<never>()
    getStatus.mockResolvedValue({ gateway_shared_with: null })
    restartGateway.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)

    const firstRestart = runGatewayRestart({ connectionId: 'owner-a', profile: 'default' })
    const secondRestart = runGatewayRestart({ connectionId: 'owner-b', profile: 'default' })
    await vi.waitFor(() => expect(restartGateway).toHaveBeenCalledTimes(2))
    expect($gatewayRestarting.get()).toBe(true)

    first.reject(new Error('owner A stopped'))
    await expect(firstRestart).resolves.toBe(false)
    expect($gatewayRestarting.get()).toBe(true)

    second.reject(new Error('owner B stopped'))
    await expect(secondRestart).resolves.toBe(false)
    expect($gatewayRestarting.get()).toBe(false)
  })

  it('does not reconnect the active gateway after the restart owner changes', async () => {
    const restart = deferred<never>()
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)
    let current = true
    getStatus.mockResolvedValue({ gateway_shared_with: null })
    restartGateway.mockReturnValue(restart.promise)

    try {
      const pending = runGatewayRestart({ connectionId: 'owner-a', profile: 'default' }, () => current)
      await vi.waitFor(() => expect(restartGateway).toHaveBeenCalled())
      current = false
      restart.reject(new Error('owner A stopped'))

      await expect(pending).resolves.toBe(false)
      expect(handler).not.toHaveBeenCalled()
    } finally {
      off()
    }
  })
})
