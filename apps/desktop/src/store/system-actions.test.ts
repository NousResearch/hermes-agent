import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getActionStatus: vi.fn(),
  notifyError: vi.fn(),
  restartGateway: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getActionStatus: mocks.getActionStatus,
  getStatus: vi.fn(),
  restartGateway: mocks.restartGateway
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => (key === 'commandCenter.gatewayRestartFailed' ? 'Messaging gateway restart failed.' : key)
}))

vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: mocks.notifyError }))

import { runGatewayRestart } from './system-actions'

describe('runGatewayRestart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.restartGateway.mockResolvedValue({ name: 'gateway-restart' })
  })

  it('surfaces the failed restart action output as the error cause', async () => {
    mocks.getActionStatus.mockResolvedValue({
      exit_code: 1,
      lines: ['starting gateway', 'Telegram token rejected: 401 Unauthorized'],
      running: false
    })

    await expect(runGatewayRestart()).resolves.toBe(false)

    expect(mocks.notifyError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Telegram token rejected: 401 Unauthorized' }),
      'Messaging gateway restart failed.'
    )
  })
})
