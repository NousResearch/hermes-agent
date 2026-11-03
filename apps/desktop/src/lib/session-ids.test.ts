import { describe, expect, it } from 'vitest'

import { storedSessionIdForNotification } from './session-ids'

describe('storedSessionIdForNotification', () => {
  it('translates a runtime id back to its stored id', () => {
    // The route is keyed by the stored id, but notifications carry the runtime
    // id. Resolving runtime -> stored keeps notification-click navigation from
    // resuming a non-existent stored session ("session not found").
    const map = new Map([['stored-abc', 'runtime-123']])

    expect(storedSessionIdForNotification('runtime-123', map)).toBe('stored-abc')
  })

  it('returns the id unchanged when no mapping is known', () => {
    // A notification for a session this window never opened may already carry a
    // stored id; let the resume/REST lookup handle it as-is.
    const map = new Map([['stored-abc', 'runtime-123']])

    expect(storedSessionIdForNotification('stored-xyz', map)).toBe('stored-xyz')
  })

  it('returns the id unchanged for an empty map', () => {
    expect(storedSessionIdForNotification('runtime-123', new Map())).toBe('runtime-123')
  })

  it('resolves the correct stored id among several sessions', () => {
    const map = new Map([
      ['stored-1', 'runtime-1'],
      ['stored-2', 'runtime-2'],
      ['stored-3', 'runtime-3']
    ])

    expect(storedSessionIdForNotification('runtime-2', map)).toBe('stored-2')
  })

  it('does not treat a stored id as a runtime id (keys are not matched)', () => {
    // The map is stored -> runtime. A value that only appears as a *key* must
    // not be rewritten, otherwise an already-stored id could be mangled.
    const map = new Map([['stored-1', 'runtime-1']])

    expect(storedSessionIdForNotification('stored-1', map)).toBe('stored-1')
  })
})
