import { describe, expect, it } from 'vitest'

import { poolTouchKeys } from './pool-touch-scope'

describe('poolTouchKeys', () => {
  it('falls back from an explicit local registry scope to its delegated bare profile', () => {
    expect(poolTouchKeys('conn:local::research')).toEqual(['conn:local::research', 'research'])
  })

  it('does not alias non-local registry scopes', () => {
    expect(poolTouchKeys('conn:homelab::research')).toEqual(['conn:homelab::research'])
  })
})
