import { describe, expect, it } from 'vitest'

import { approvalReplaySessionId, gatewayEventRequiresSessionId, resolveGatewayEventSessionId } from './gateway-events'

describe('gateway event routing', () => {
  it('rehydrates pending approvals on reconnect ready and resumed session info', () => {
    expect(approvalReplaySessionId('gateway.ready', 'active-1', null)).toBe('active-1')
    expect(approvalReplaySessionId('session.info', 'active-1', 'routed-1')).toBe('routed-1')
    expect(approvalReplaySessionId('message.delta', 'active-1', 'routed-1')).toBeNull()
  })

  it('does not replay against an active runtime the gateway already reported gone', () => {
    const isGone = (sid: string) => sid === 'dead-1'

    // Unscoped fan-out attributed to a dead active session: skip.
    expect(approvalReplaySessionId('session.info', 'dead-1', 'dead-1', { explicit: false, isGone })).toBeNull()
    expect(approvalReplaySessionId('gateway.ready', 'dead-1', null, { explicit: false, isGone })).toBeNull()
    // A live active session still replays.
    expect(approvalReplaySessionId('session.info', 'live-1', 'live-1', { explicit: false, isGone })).toBe('live-1')
    // An explicitly scoped frame is the runtime speaking for itself — never skipped.
    expect(approvalReplaySessionId('session.info', 'dead-1', 'dead-1', { explicit: true, isGone })).toBe('dead-1')
  })

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
      pinned: false,
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
      pinned: true,
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
      pinned: true,
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
      pinned: false,
      sessionId: 'session-b'
    })
  })

  it('attributes an unpinned stream event to the active session without the pin flag', () => {
    // A late straggler (no pin left after the previous turn completed) falls
    // back to the active session. The handler drops this case when the target
    // session has no live turn — the straggler belongs to a turn that already
    // ended elsewhere (#43142 family).
    const routed = resolveGatewayEventSessionId({
      activeSessionId: 'session-b',
      eventType: 'thinking.delta',
      explicitSessionId: '',
      unscopedStreamSessionId: null
    })

    expect(routed).toEqual({
      drop: false,
      nextUnscopedStreamSessionId: null,
      pinned: false,
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
      pinned: true,
      sessionId: 'session-a'
    })
  })

  type UnscopedRouteState = {
    activeSessionId: null | string
    pinnedSessionHasLiveTurn?: boolean
    unscopedStreamContested?: boolean
    unscopedStreamSessionId: null | string
  }

  function applyUnscoped(eventType: string, state: UnscopedRouteState) {
    const route = resolveGatewayEventSessionId({
      activeSessionId: state.activeSessionId,
      eventType,
      explicitSessionId: '',
      pinnedSessionHasLiveTurn: state.pinnedSessionHasLiveTurn,
      unscopedStreamContested: state.unscopedStreamContested,
      unscopedStreamSessionId: state.unscopedStreamSessionId
    })

    return {
      route,
      state: {
        ...state,
        unscopedStreamContested: route.nextUnscopedStreamContested ?? false,
        unscopedStreamSessionId: route.nextUnscopedStreamSessionId
      }
    }
  }

  it('does not steal the unscoped stream pin onto a focused session that started while the pin is live (#108045)', () => {
    let state: UnscopedRouteState = {
      activeSessionId: 'session-a',
      pinnedSessionHasLiveTurn: true,
      unscopedStreamSessionId: null
    }

    ;({ state } = applyUnscoped('message.start', state))
    expect(state.unscopedStreamSessionId).toBe('session-a')

    state = { ...state, activeSessionId: 'session-b', pinnedSessionHasLiveTurn: true }

    const startB = applyUnscoped('message.start', state)

    expect(startB.route).toEqual({
      drop: false,
      nextUnscopedStreamContested: true,
      nextUnscopedStreamSessionId: 'session-a',
      pinned: false,
      sessionId: 'session-b'
    })

    const delta = applyUnscoped('message.delta', startB.state)

    expect(delta.route.drop).toBe(true)
    expect(delta.route.sessionId).toBeNull()
    expect(delta.route.sessionId).not.toBe('session-b')

    const thinking = applyUnscoped('thinking.delta', delta.state)

    expect(thinking.route.drop).toBe(true)
    expect(thinking.route.sessionId).toBeNull()
  })

  it('does not steal the pin across two live unscoped starts (slot thrash)', () => {
    let state: UnscopedRouteState = {
      activeSessionId: 'session-a',
      pinnedSessionHasLiveTurn: true,
      unscopedStreamSessionId: null
    }

    ;({ state } = applyUnscoped('message.start', state))

    state = { ...state, activeSessionId: 'session-b', pinnedSessionHasLiveTurn: true }

    const startB = applyUnscoped('message.start', state)

    expect(startB.route.sessionId).toBe('session-b')
    expect(startB.route.nextUnscopedStreamSessionId).toBe('session-a')
    expect(startB.route.nextUnscopedStreamContested).toBe(true)
    expect(startB.route.drop).toBe(false)

    const startC = applyUnscoped('message.start', { ...startB.state, activeSessionId: 'session-c' })

    expect(startC.route.sessionId).toBe('session-c')
    expect(startC.route.nextUnscopedStreamSessionId).toBe('session-a')
    expect(startC.route.nextUnscopedStreamContested).toBe(true)

    const delta = applyUnscoped('message.delta', startC.state)

    expect(delta.route.drop).toBe(true)
    expect(delta.route.sessionId).toBeNull()
  })

  it('drops an unscoped stream end while contested and clears the pin', () => {
    let state: UnscopedRouteState = {
      activeSessionId: 'session-a',
      pinnedSessionHasLiveTurn: true,
      unscopedStreamSessionId: null
    }

    ;({ state } = applyUnscoped('message.start', state))
    state = { ...state, activeSessionId: 'session-b', pinnedSessionHasLiveTurn: true }

    const startB = applyUnscoped('message.start', state)
    const completed = applyUnscoped('message.complete', startB.state)

    expect(completed.route.drop).toBe(true)
    expect(completed.route.sessionId).toBeNull()
    expect(completed.route.nextUnscopedStreamSessionId).toBeNull()
    expect(completed.route.nextUnscopedStreamContested ?? false).toBe(false)

    const errored = applyUnscoped('error', startB.state)

    expect(errored.route.drop).toBe(true)
    expect(errored.route.sessionId).toBeNull()
    expect(errored.route.nextUnscopedStreamSessionId).toBeNull()
    expect(errored.route.nextUnscopedStreamContested ?? false).toBe(false)
  })
})
