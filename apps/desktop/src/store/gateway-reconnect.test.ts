import { afterEach, describe, expect, it, vi } from 'vitest'

import { reconnectGateway, registerGatewayReconnect } from './gateway-reconnect'

const disposers: Array<() => void> = []

afterEach(() => {
  while (disposers.length > 0) {
    disposers.pop()?.()
  }
})

describe('gateway reconnect controller', () => {
  it('coalesces repeated requests onto one active reconnect', async () => {
    let finish: (() => void) | undefined

    const handler = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>(resolve => {
            finish = resolve
          })
      )
      .mockResolvedValueOnce(undefined)

    disposers.push(registerGatewayReconnect(handler))

    const first = reconnectGateway()
    const second = reconnectGateway()

    expect(second).toBe(first)
    await Promise.resolve()
    expect(handler).toHaveBeenCalledTimes(1)

    finish?.()
    await first

    await reconnectGateway()
    expect(handler).toHaveBeenCalledTimes(2)
  })

  it('only lets the current registration remove itself', async () => {
    const stale = vi.fn()
    const current = vi.fn()
    const disposeStale = registerGatewayReconnect(stale)
    disposers.push(registerGatewayReconnect(current))

    disposeStale()
    await reconnectGateway()

    expect(stale).not.toHaveBeenCalled()
    expect(current).toHaveBeenCalledOnce()
  })

  it('rejects when the gateway boot owner is not mounted', async () => {
    await expect(reconnectGateway()).rejects.toThrow('Gateway reconnect is unavailable')
  })
})
