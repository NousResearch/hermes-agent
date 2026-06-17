import { describe, expect, it } from 'vitest'

import { gatewayEventRequiresSessionId, resolveGatewayEventSessionId } from './gateway-events'

describe('gateway event routing', () => {
  it('drops only unscoped subagent events (genuinely background work)', () => {
    expect(gatewayEventRequiresSessionId('subagent.progress')).toBe(true)
    expect(gatewayEventRequiresSessionId('subagent.start')).toBe(true)
  })

  it('attributes unscoped foreground turn events to the active chat', () => {
    // These must NOT be dropped when unscoped — they are the focused turn's own
    // output, and dropping them loses the live response until a refetch (#42178).
    expect(gatewayEventRequiresSessionId('message.delta')).toBe(false)
    expect(gatewayEventRequiresSessionId('message.complete')).toBe(false)
    expect(gatewayEventRequiresSessionId('reasoning.delta')).toBe(false)
    expect(gatewayEventRequiresSessionId('tool.start')).toBe(false)
    expect(gatewayEventRequiresSessionId('approval.request')).toBe(false)
  })

  it('allows global events to remain unscoped', () => {
    expect(gatewayEventRequiresSessionId('gateway.ready')).toBe(false)
    expect(gatewayEventRequiresSessionId('preview.restart.progress')).toBe(false)
    expect(gatewayEventRequiresSessionId('session.info')).toBe(false)
    expect(gatewayEventRequiresSessionId(undefined)).toBe(false)
  })

  it('routes unscoped events to the sole busy session', () => {
    const states = new Map([
      ['bg-runtime', { busy: true }],
      ['fg-runtime', { busy: false }]
    ])

    expect(resolveGatewayEventSessionId('', 'fg-runtime', states)).toBe('bg-runtime')
  })

  it('prefers the active session when multiple sessions are busy', () => {
    const states = new Map([
      ['bg-runtime', { busy: true }],
      ['fg-runtime', { busy: true }]
    ])

    expect(resolveGatewayEventSessionId('', 'fg-runtime', states)).toBe('fg-runtime')
  })

  it('drops ambiguous unscoped events when several sessions are busy and active is quiet', () => {
    const states = new Map([
      ['a-runtime', { busy: true }],
      ['b-runtime', { awaitingResponse: true }]
    ])

    expect(resolveGatewayEventSessionId('', 'quiet-runtime', states)).toBeNull()
  })

  it('keeps explicit session ids authoritative', () => {
    const states = new Map([['bg-runtime', { busy: true }]])

    expect(resolveGatewayEventSessionId('explicit-runtime', 'fg-runtime', states)).toBe('explicit-runtime')
  })
})
