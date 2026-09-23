import { describe, expect, it } from 'vitest'

import { shouldRefreshGatewayScope } from './gateway-scope-refresh'

describe('shouldRefreshGatewayScope', () => {
  it('waits for the initial gateway to open before accepting its scope', () => {
    expect(shouldRefreshGatewayScope(null, 'local\0default', 'connecting')).toBe(false)
    expect(shouldRefreshGatewayScope(null, 'local\0default', 'open')).toBe(true)
  })

  it('refreshes each open scope once', () => {
    expect(shouldRefreshGatewayScope('local\0default', 'local\0default', 'open')).toBe(false)
    expect(shouldRefreshGatewayScope('local\0default', 'local\0research', 'open')).toBe(true)
  })
})