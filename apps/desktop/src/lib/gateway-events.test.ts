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
      unscopedStreamSessionIds: []
    })

    expect(started).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-a'],
      sessionId: 'session-a'
    })

    const delta = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionIds: started.nextUnscopedStreamSessionIds
    })

    expect(delta).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-a'],
      sessionId: 'session-a'
    })

    const completed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: '',
      unscopedStreamSessionIds: delta.nextUnscopedStreamSessionIds
    })

    expect(completed).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: [],
      sessionId: 'session-a'
    })
  })

  it('routes a new unscoped stream start to the currently active session', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.start',
      explicitSessionId: '',
      unscopedStreamSessionIds: ['session-a']
    })

    // Session B owns its own start, but A's stream is still running and keeps
    // its pin — the second start adds, it does not take over.
    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-a', 'session-b'],
      sessionId: 'session-b'
    })
  })

  it('keeps explicit events scoped and clears a matching pinned stream on completion', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: 'session-a',
      unscopedStreamSessionIds: ['session-a']
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: [],
      sessionId: 'session-a'
    })
  })

  it('retires only the completing stream when several run concurrently', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: 'session-a',
      unscopedStreamSessionIds: ['session-a', 'session-b']
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-b'],
      sessionId: 'session-a'
    })
  })

  it("does not let a second chat's stream steal the first chat's unscoped events", () => {
    // The #46194 / #62823 race. A single shared pin was overwritten by B's
    // message.start, so every later unscoped event from A resolved to B and
    // painted A's output onto B's transcript.
    const aStarted = resolveGatewayEventSessionId({
      activeSessionId: 'session-a',
      eventType: 'message.start',
      explicitSessionId: '',
      unscopedStreamSessionIds: []
    })

    const bStarted = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.start',
      explicitSessionId: '',
      unscopedStreamSessionIds: aStarted.nextUnscopedStreamSessionIds
    })

    expect(bStarted.nextUnscopedStreamSessionIds).toEqual(['session-a', 'session-b'])

    // A's stream completes while B is focused and still streaming. Before the
    // per-stream pins this resolved to 'session-b'.
    const aCompleted = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.complete',
      explicitSessionId: 'session-a',
      unscopedStreamSessionIds: bStarted.nextUnscopedStreamSessionIds
    })

    expect(aCompleted.sessionId).toBe('session-a')
    expect(aCompleted.nextUnscopedStreamSessionIds).toEqual(['session-b'])

    // B's own unscoped delta still lands on B, unambiguously.
    const bDelta = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionIds: aCompleted.nextUnscopedStreamSessionIds
    })

    expect(bDelta).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-b'],
      sessionId: 'session-b'
    })
  })

  it('attributes an ambiguous unscoped delta to the focused chat when it is streaming', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionIds: ['session-a', 'session-b']
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-a', 'session-b'],
      sessionId: 'session-b'
    })
  })

  it('drops an unscoped delta it cannot attribute to any of several live streams', () => {
    // Two background streams, focused chat idle: nothing in the event says
    // which stream it came from, and guessing is what grafts A's output onto B.
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-c',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionIds: ['session-a', 'session-b']
    })

    expect(routed).toEqual({
      drop: true,
      nextUnscopedStreamSessionIds: ['session-a', 'session-b'],
      sessionId: null
    })
  })

  it('leaves single-stream and no-stream routing unchanged', () => {
    // #70376 owns tightening the no-pin fallback; this change must not move it.
    const lateDelta = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'message.delta',
      explicitSessionId: '',
      unscopedStreamSessionIds: []
    })

    expect(lateDelta).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: [],
      sessionId: 'session-b'
    })

    const nonStreamEvent = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'session.info',
      explicitSessionId: '',
      unscopedStreamSessionIds: ['session-a']
    })

    expect(nonStreamEvent).toEqual({
      drop: false,
      nextUnscopedStreamSessionIds: ['session-a'],
      sessionId: 'session-b'
    })
  })

  it('still drops unscoped subagent events without disturbing live pins', () => {
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'subagent.progress',
      explicitSessionId: '',
      unscopedStreamSessionIds: ['session-a']
    })

    expect(routed).toEqual({
      drop: true,
      nextUnscopedStreamSessionIds: ['session-a'],
      sessionId: null
    })
  })
})
