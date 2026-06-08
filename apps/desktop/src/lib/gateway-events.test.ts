import { describe, expect, it } from 'vitest'

import { gatewayEventRequiresSessionId } from './gateway-events'

describe('gateway event routing', () => {
  it('requires explicit session ids for async session-scoped events', () => {
    expect(gatewayEventRequiresSessionId('message.delta')).toBe(true)
    expect(gatewayEventRequiresSessionId('tool.start')).toBe(true)
    expect(gatewayEventRequiresSessionId('subagent.progress')).toBe(true)
    expect(gatewayEventRequiresSessionId('approval.request')).toBe(true)
  })

  it('allows global events to remain unscoped', () => {
    expect(gatewayEventRequiresSessionId('gateway.ready')).toBe(false)
    expect(gatewayEventRequiresSessionId('preview.restart.progress')).toBe(false)
    expect(gatewayEventRequiresSessionId('session.info')).toBe(false)
    expect(gatewayEventRequiresSessionId(undefined)).toBe(false)
  })
})
