import { describe, expect, it, vi } from 'vitest'

import { GatewayReauthRequiredError } from '@/lib/gateway-ws-url'

import { connectInitialGateway } from './use-gateway-boot'

const never = () => false

describe('connectInitialGateway (initial-boot retry)', () => {
  it('retries past transient failures and resolves once a connect succeeds', async () => {
    const connect = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new Error('1006'))
      .mockRejectedValueOnce(new Error('1006'))
      .mockResolvedValueOnce(undefined)

    await expect(
      connectInitialGateway({ connect, delayMs: 0, isCancelled: never })
    ).resolves.toBeUndefined()
    expect(connect).toHaveBeenCalledTimes(3)
  })

  it('throws the last error after exhausting all attempts', async () => {
    const connect = vi.fn<() => Promise<void>>().mockRejectedValue(new Error('backend cold'))

    await expect(
      connectInitialGateway({ attempts: 3, connect, delayMs: 0, isCancelled: never })
    ).rejects.toThrow('backend cold')
    expect(connect).toHaveBeenCalledTimes(3)
  })

  it('fails fast on reauth errors without retrying', async () => {
    const connect = vi
      .fn<() => Promise<void>>()
      .mockRejectedValue(new GatewayReauthRequiredError('sign in again'))

    await expect(
      connectInitialGateway({ connect, delayMs: 0, isCancelled: never })
    ).rejects.toBeInstanceOf(GatewayReauthRequiredError)
    expect(connect).toHaveBeenCalledTimes(1)
  })

  it('stops retrying once cancelled (component unmounted)', async () => {
    let cancelled = false
    const connect = vi.fn<() => Promise<void>>().mockImplementation(() => {
      cancelled = true

      return Promise.reject(new Error('1006'))
    })

    await expect(
      connectInitialGateway({ connect, delayMs: 0, isCancelled: () => cancelled })
    ).rejects.toThrow('1006')
    expect(connect).toHaveBeenCalledTimes(1)
  })
})
