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
    expect(gatewayEventRequiresSessionId('message.interim')).toBe(false)
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

  it('keeps unscoped stream events pinned to the session that started them', () => {
    const started = resolveGatewayEventSessionId({
      activeSessionId: 'session-a',
      eventType: 'message.start',
      explicitSessionId: '',
      unscopedStreamSessionId: null
    })

    expect(started).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: 'session-a',
      sessionId: 'session-a'
    })

    const delta = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionId: started.nextUnscopedStreamSessionId
    })

    expect(delta).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: 'session-a',
      sessionId: 'session-a'
    })

    const completed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: '',
      unscopedStreamSessionId: delta.nextUnscopedStreamSessionId
    })

    expect(completed).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: null,
      sessionId: 'session-a'
    })
  })

  it('routes a new unscoped stream start to the currently active session', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.start',
      explicitSessionId: '',
      unscopedStreamSessionId: 'session-a'
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: 'session-b',
      sessionId: 'session-b'
    })
  })

  it('keeps explicit events scoped and clears a matching pinned stream on completion', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: 'session-a',
      unscopedStreamSessionId: 'session-a'
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: null,
      sessionId: 'session-a'
    })
  })
})
