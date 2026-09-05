import { describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { host } from '@/sdk'
import { $connection, setConnection } from '@/store/session'

/**
 * #101195 regression: the forever-"Waiting for the gateway connection…" card
 * hid the endpoint it was retrying. A user whose active connection pointed at
 * the wrong host/port (the reporter's OpenClaw port collision) saw a silent
 * banner with no way to tell WHERE Hermes was trying to connect.
 *
 * The fix exposes the resolved connection descriptor (HermesConnection —
 * published by boot/reconnect BEFORE the WebSocket opens) through
 * `host.state.gatewayConnection` so a stuck-connection surface can name the
 * endpoint.
 */

describe('host.state.gatewayConnection (#101195)', () => {
  it('mirrors $connection — the descriptor is available while gateway is down', () => {
    // The slice is a readonly projection of the same $connection atom the
    // boot paths publish before gateway.connect(wsUrl) — the property that
    // makes the endpoint nameable during a stuck retry, not just after open.
    expect(host.state.gatewayConnection).toBeDefined()

    // With no connection published (fresh boot / tests), the slice reads
    // null — the roster card hides the endpoint row, matching pre-fix
    // behavior when nothing is known.
    expect(host.state.gatewayConnection.get()).toBeNull()
  })

  it('projects the full descriptor when a connection is published', () => {
    const conn = {
      baseUrl: 'http://127.0.0.1:53150',
      wsUrl: 'ws://127.0.0.1:53150/api/ws?token=secret',
      mode: 'local',
      isFullscreen: false,
      nativeOverlayWidth: 0,
      token: 'secret',
      windowButtonPosition: null,
      logs: []
    } as HermesConnection

    const prev = $connection.get()
    setConnection(conn)

    expect(host.state.gatewayConnection.get()?.baseUrl).toBe('http://127.0.0.1:53150')
    expect(host.state.gatewayConnection.get()?.wsUrl).toContain('ws://127.0.0.1:53150')
    expect(host.state.gatewayConnection.get()?.mode).toBe('local')

    setConnection(prev)
  })
})
