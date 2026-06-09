import { describe, expect, it } from 'vitest'

import { reconcileOrderIds } from './order'

describe('reconcileOrderIds', () => {
  it('does not seed a manual order from the natural session order', () => {
    expect(reconcileOrderIds(['newest', 'older'], [])).toEqual([])
  })

  it('clears an old auto-seeded natural order when new sessions arrive', () => {
    expect(reconcileOrderIds(['newest', 'older', 'oldest'], ['older', 'oldest'])).toEqual([])
  })

  it('keeps manual order and appends newly loaded ids', () => {
    expect(reconcileOrderIds(['newest', 'older', 'oldest'], ['oldest', 'older'])).toEqual([
      'oldest',
      'older',
      'newest'
    ])
  })
})
