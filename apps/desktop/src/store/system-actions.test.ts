// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { translateNow } from '@/i18n'

import { runGatewayRestart } from './system-actions'

const restartGateway = vi.fn()
const getActionStatus = vi.fn()
const getStatus = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  getActionStatus: (...args: unknown[]) => getActionStatus(...args),
  getStatus: (...args: unknown[]) => getStatus(...args),
  restartGateway: (...args: unknown[]) => restartGateway(...args)
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notify(...args),
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

beforeEach(() => {
  vi.clearAllMocks()
  // Standalone gateway: no shared-multiplexer confirm dialog.
  getStatus.mockResolvedValue({})
  restartGateway.mockResolvedValue({ name: 'gateway-restart' })
})

describe('runGatewayRestart failure cause (#120641)', () => {
  it('attaches the exit code to the failure instead of repeating the fallback', async () => {
    // awaitAction used to throw the fallback sentence itself, so the toast
    // showed "Gateway restart failed." as title AND message with zero cause.
    // The thrown error must carry what the backend reported (exit code).
    getActionStatus.mockResolvedValue({ running: false, exit_code: 1 })

    const ok = await runGatewayRestart()

    expect(ok).toBe(false)
    expect(notifyError).toHaveBeenCalledTimes(1)
    const [err, fallback] = notifyError.mock.calls[0] as [unknown, string]
    expect(fallback).toBe(translateNow('commandCenter.gatewayRestartFailed'))
    const message = err instanceof Error ? err.message : String(err)
    expect(message).toContain('1')
    expect(message).not.toBe(fallback)
  })

  it('resolves true and stays silent on a clean restart', async () => {
    getActionStatus.mockResolvedValue({ running: false, exit_code: 0 })

    const ok = await runGatewayRestart()

    expect(ok).toBe(true)
    expect(notifyError).not.toHaveBeenCalled()
  })
})
