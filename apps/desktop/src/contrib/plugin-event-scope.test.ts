import { describe, expect, it } from 'vitest'

import { emitGatewayEvent, onGatewayEvent, withEventSubscriptionScope } from './events'
import { createPluginContext } from './plugin'

// Regression for #112366: runtime plugins leaked gateway-event listeners on
// every hot reload — a bare `host.onEvent` inside register() survived unload,
// so one relay marker opened N duplicate sessions.
describe('plugin event subscriptions unload with the plugin', () => {
  const ping = () => emitGatewayEvent({ type: 'gateway.reconnecting', payload: { attempt: 1 } })

  it('a bare onEvent during register is disposed by the loader unload', () => {
    const disposers: Array<() => void> = []
    const seen: unknown[] = []

    // What the loaders do around plugin.register():
    withEventSubscriptionScope(
      dispose => disposers.push(dispose),
      () => void onGatewayEvent('gateway.reconnecting', event => seen.push(event))
    )

    ping()
    expect(seen).toHaveLength(1)

    // The loader's unload: every tracked disposer runs.
    disposers.forEach(dispose => dispose())
    ping()
    expect(seen).toHaveLength(1)
  })

  it('ctx.onEvent is tracked like every other registration', () => {
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('demo', dispose => disposers.push(dispose))
    const seen: unknown[] = []

    ctx.onEvent('gateway.reconnecting', event => seen.push(event))
    ping()
    expect(seen).toHaveLength(1)

    disposers.forEach(dispose => dispose())
    ping()
    expect(seen).toHaveLength(1)
  })

  it('subscriptions outside a register window stay caller-owned', () => {
    const seen: unknown[] = []
    const dispose = onGatewayEvent('gateway.reconnecting', event => seen.push(event))

    ping()
    expect(seen).toHaveLength(1)

    // No scope captured it, so it still fires until the caller disposes.
    ping()
    expect(seen).toHaveLength(2)

    dispose()
    ping()
    expect(seen).toHaveLength(2)
  })
})
