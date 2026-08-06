import { describe, expect, it } from 'vitest'

import examplePlugin from './example/plugin'
import gatewayPillPlugin from './gateway-pill/plugin'

describe('bundled demo plugin defaults', () => {
  it('keeps dogfood/demo plugins inventoried but opt-in by default', () => {
    expect(examplePlugin.id).toBe('example')
    expect(examplePlugin.defaultEnabled).toBe(false)
    expect(gatewayPillPlugin.id).toBe('gateway-pill')
    expect(gatewayPillPlugin.defaultEnabled).toBe(false)
  })
})
