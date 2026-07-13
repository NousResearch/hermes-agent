import { describe, expect, it } from 'vitest'

import { contextCapacityClassName } from './statusbar'

describe('contextCapacityClassName', () => {
  it('keeps empty context neutral and reserves color for meaningful capacity bands', () => {
    expect(contextCapacityClassName(0)).toContain('muted')
    expect(contextCapacityClassName(24)).toContain('--ui-green')
    expect(contextCapacityClassName(50)).toContain('--ui-yellow')
    expect(contextCapacityClassName(80)).toContain('--ui-orange')
    expect(contextCapacityClassName(95)).toContain('--ui-red')
  })
})
