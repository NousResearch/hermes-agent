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

import { $gatewayRestarting, runGatewayRestart } from './system-actions'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}

describe('runGatewayRestart owner lifetime', () => {
  beforeEach(() => {
    getStatus.mockResolvedValue({ gateway_shared_with: ['default', 'research'] })
    getActionStatus.mockResolvedValue({ running: false, exit_code: 0 })
    restartGateway.mockResolvedValue({ name: 'gateway-restart' })
  })

  afterEach(() => {
    vi.clearAllMocks()
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
})
