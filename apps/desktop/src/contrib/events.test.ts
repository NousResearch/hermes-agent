import { describe, expect, it, vi } from 'vitest'

import { emitGatewayEvent, onGatewayEvent } from './events'

describe('onGatewayEvent', () => {
  it('keeps the legacy direct subscription and disposer behavior outside plugin registration', () => {
    const listener = vi.fn()
    const dispose = onGatewayEvent('gateway.ready', listener)

    emitGatewayEvent({ type: 'gateway.ready' } as never)
    expect(listener).toHaveBeenCalledTimes(1)

    dispose()
    emitGatewayEvent({ type: 'gateway.ready' } as never)
    expect(listener).toHaveBeenCalledTimes(1)
  })
})
