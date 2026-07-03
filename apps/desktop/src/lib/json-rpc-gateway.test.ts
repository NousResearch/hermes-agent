import { describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from '@hermes/shared'

describe('JsonRpcGatewayClient connect', () => {
  it('resets state after a synchronous socket construction failure', async () => {
    const socketFactory = vi
      .fn()
      .mockImplementationOnce(() => {
        throw new Error('bad socket url')
      })
      .mockImplementationOnce(() => {
        throw new Error('retry reached factory')
      })
    const client = new JsonRpcGatewayClient({ socketFactory })

    await expect(client.connect('wss://example.invalid/ws')).rejects.toThrow('bad socket url')

    expect(client.connectionState).toBe('error')

    await expect(client.connect('wss://example.invalid/ws')).rejects.toThrow('retry reached factory')
    expect(socketFactory).toHaveBeenCalledTimes(2)
  })
})
