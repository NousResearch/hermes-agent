import { describe, expect, it } from 'vitest'

import { isRouteSessionMismatch } from './route-session-state'

describe('isRouteSessionMismatch', () => {
  it('keeps the composer mounted when auto-compression rotates root to tip', () => {
    const sessions = [{ id: 'tip-2', _lineage_root_id: 'root-1' }]

    expect(isRouteSessionMismatch('root-1', 'tip-2', sessions)).toBe(false)
    expect(isRouteSessionMismatch('tip-2', 'root-1', sessions)).toBe(false)
  })

  it('still suppresses the previous chat while a genuinely different route loads', () => {
    const sessions = [{ id: 'selected', _lineage_root_id: null }]

    expect(isRouteSessionMismatch('next-session', 'selected', sessions)).toBe(true)
    expect(isRouteSessionMismatch('next-session', null, sessions)).toBe(true)
  })

  it('does not require a session row when the selected and routed ids already agree', () => {
    expect(isRouteSessionMismatch('same', 'same', [])).toBe(false)
    expect(isRouteSessionMismatch(null, 'same', [])).toBe(false)
  })
})
