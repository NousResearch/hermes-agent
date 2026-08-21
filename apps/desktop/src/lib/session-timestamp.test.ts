import { describe, expect, it } from 'vitest'

import { dateRepresentableUnixSeconds, sessionListRecencySeconds } from './session-timestamp'

describe('session timestamp admission', () => {
  it('rejects malformed, non-finite, and out-of-range values', () => {
    for (const value of [NaN, Infinity, -Infinity, 8_640_000_000_001, 'not-a-timestamp', null]) {
      expect(dateRepresentableUnixSeconds(value)).toBeNull()
    }
  })

  it('uses a valid last_active and preserves zero-as-unset fallback', () => {
    expect(sessionListRecencySeconds({ last_active: 20, started_at: 10 })).toBe(20)
    expect(sessionListRecencySeconds({ last_active: 0, started_at: 10 })).toBe(10)
    expect(sessionListRecencySeconds({ last_active: Infinity, started_at: 10 })).toBe(10)
    expect(sessionListRecencySeconds({ last_active: Infinity, started_at: Infinity })).toBe(0)
  })
})
