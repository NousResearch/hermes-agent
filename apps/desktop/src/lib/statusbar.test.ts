import { describe, expect, it } from 'vitest'

import { contextCapacityClassName } from './statusbar'

describe('contextCapacityClassName', () => {
  it('keeps zero muted and escalates only the capacity meter color', () => {
    expect(contextCapacityClassName(0)).toContain('text-muted-foreground')
    expect(contextCapacityClassName(25)).toContain('--ui-green')
    expect(contextCapacityClassName(50)).toContain('--ui-yellow')
    expect(contextCapacityClassName(80)).toContain('--ui-orange')
    expect(contextCapacityClassName(95)).toContain('--ui-red')
  })
})
