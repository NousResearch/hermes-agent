import { describe, expect, it } from 'vitest'

import { visibleOrderingOptions } from './filter-menu'

describe('visibleOrderingOptions', () => {
  it('always exposes Manual so keyboard and menu users can enter reorder mode', () => {
    expect(visibleOrderingOptions(false, 'updated').map(option => option.id)).toContain('manual')
  })

  it('keeps Cost conditional unless it is the active ordering', () => {
    expect(visibleOrderingOptions(false, 'updated').map(option => option.id)).not.toContain('cost')
    expect(visibleOrderingOptions(false, 'cost').map(option => option.id)).toContain('cost')
  })
})
