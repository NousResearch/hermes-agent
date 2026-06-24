import { describe, expect, it } from 'vitest'

import { gatewayEventRequiresSessionId } from './gateway-events'

describe('gateway event routing', () => {
  it('drops unscoped subagent events (genuinely background work)', () => {
    expect(gatewayEventRequiresSessionId('subagent.progress')).toBe(true)
    expect(gatewayEventRequiresSessionId('subagent.start')).toBe(true)
  })

  it('drops unscoped session-scoped events (message, tool, reasoning, etc.)', () => {
    // Fixes #49106 / #47709: these are session-scoped and must carry an
    // explicit session_id. When the gateway fails to stamp one, dropping
    // is safer than attributing to whichever session is focused.
    expect(gatewayEventRequiresSessionId('message.delta')).toBe(true)
    expect(gatewayEventRequiresSessionId('message.complete')).toBe(true)
    expect(gatewayEventRequiresSessionId('reasoning.delta')).toBe(true)
    expect(gatewayEventRequiresSessionId('tool.start')).toBe(true)
    expect(gatewayEventRequiresSessionId('approval.request')).toBe(true)
    expect(gatewayEventRequiresSessionId('session.info')).toBe(true)
    expect(gatewayEventRequiresSessionId('preview.restart.progress')).toBe(true)
  })

  it('allows global broadcasts to remain unscoped', () => {
    expect(gatewayEventRequiresSessionId('gateway.ready')).toBe(false)
  })

  it('requires session_id for unknown event types (defensive)', () => {
    expect(gatewayEventRequiresSessionId(undefined)).toBe(true)
    expect(gatewayEventRequiresSessionId('')).toBe(true)
  })
})
