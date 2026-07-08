import { describe, expect, it } from 'vitest'

import { gatewayEventRequiresSessionId } from './gateway-events'

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

  it('documents the Race 3 fix: session.info state patches require explicitSid', () => {
    // gatewayEventRequiresSessionId('session.info') is still false — unscoped
    // session.info events are NOT dropped (they carry the active turn's state).
    // The Race 3 fix is in gateway-event.ts: the per-session state-patch
    // (updateSessionState) is guarded by `explicitSid` so an unscoped
    // session.info can't misattribute a background session's model/cwd/branch
    // onto the active session during a switch.
    //
    // This test documents the invariant: the function-level gate stays open
    // for session.info, but the state-patch path in the event handler checks
    // explicitSid separately.
    expect(gatewayEventRequiresSessionId('session.info')).toBe(false)
  })
})
