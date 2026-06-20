import { describe, expect, it } from 'vitest'

import { gatewayEventRequiresSessionId } from './gateway-events'

describe('gateway event routing', () => {
  it('drops unscoped session-scoped events instead of leaking them into the focused chat', () => {
    expect(gatewayEventRequiresSessionId('subagent.progress')).toBe(true)
    expect(gatewayEventRequiresSessionId('subagent.start')).toBe(true)
    expect(gatewayEventRequiresSessionId('message.delta')).toBe(true)
    expect(gatewayEventRequiresSessionId('message.complete')).toBe(true)
    expect(gatewayEventRequiresSessionId('reasoning.delta')).toBe(true)
    expect(gatewayEventRequiresSessionId('tool.start')).toBe(true)
    expect(gatewayEventRequiresSessionId('tool.complete')).toBe(true)
    expect(gatewayEventRequiresSessionId('approval.request')).toBe(true)
    expect(gatewayEventRequiresSessionId('clarify.request')).toBe(true)
    expect(gatewayEventRequiresSessionId('sudo.request')).toBe(true)
    expect(gatewayEventRequiresSessionId('secret.request')).toBe(true)
    expect(gatewayEventRequiresSessionId('status.update')).toBe(true)
    expect(gatewayEventRequiresSessionId('review.summary')).toBe(true)
  })

  it('allows global events to remain unscoped', () => {
    expect(gatewayEventRequiresSessionId('gateway.ready')).toBe(false)
    expect(gatewayEventRequiresSessionId('error')).toBe(false)
    expect(gatewayEventRequiresSessionId('preview.restart.progress')).toBe(false)
    expect(gatewayEventRequiresSessionId('session.info')).toBe(false)
    expect(gatewayEventRequiresSessionId(undefined)).toBe(false)
  })
})
