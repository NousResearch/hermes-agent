import { describe, expect, it } from 'vitest'

import {
  armUpdateContinuation,
  clearUpdateContinuation,
  continuationSessionId,
  readUpdateContinuation,
  type UpdateContinuationStorage
} from './update-continuation'

function memoryStorage(): UpdateContinuationStorage {
  const values = new Map<string, string>()

  return {
    getItem: key => values.get(key) ?? null,
    removeItem: key => values.delete(key),
    setItem: (key, value) => values.set(key, value)
  }
}

describe('update continuation handoff', () => {
  it('arms only for an in-flight turn and prefers the durable stored id', () => {
    expect(
      continuationSessionId({ activeStoredSessionId: 'stored-1', selectedStoredSessionId: 'selected-1', busy: true })
    ).toBe('stored-1')
    expect(
      continuationSessionId({ activeStoredSessionId: null, selectedStoredSessionId: 'selected-1', busy: false })
    ).toBeNull()
  })

  it('keeps the handoff durable until delivery is explicitly acknowledged', () => {
    const storage = memoryStorage()

    armUpdateContinuation('stored-1', { now: 1_000, storage })

    expect(readUpdateContinuation({ now: 2_000, storage })?.sessionId).toBe('stored-1')
    expect(readUpdateContinuation({ now: 2_001, storage })?.sessionId).toBe('stored-1')
    clearUpdateContinuation(storage)
    expect(readUpdateContinuation({ now: 2_002, storage })).toBeNull()
  })

  it('drops stale handoffs instead of reviving an old task', () => {
    const storage = memoryStorage()

    armUpdateContinuation('stored-1', { now: 1_000, storage })

    expect(readUpdateContinuation({ maxAgeMs: 500, now: 2_000, storage })).toBeNull()
  })
})
