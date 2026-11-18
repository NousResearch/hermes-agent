import { describe, expect, it } from 'vitest'

import { threadMessageIndex, threadUserOrdinal } from './thread-message-index'

const rows = [
  { id: 'u1', role: 'user' },
  { id: 'a1', role: 'assistant' },
  { id: 'u2', role: 'user' },
  { id: 'a2', role: 'assistant' }
]

describe('thread message index', () => {
  it('finds a row index and a user ordinal, or reports none', () => {
    expect(threadMessageIndex(rows, 'a2')).toBe(3)
    expect(threadMessageIndex(rows, 'missing')).toBe(-1)
    expect(threadUserOrdinal(rows, 'u2')).toBe(1)
    expect(threadUserOrdinal(rows, 'a1')).toBeNull()
  })

  it('is keyed by array identity, so a new transcript array is re-indexed', () => {
    expect(threadMessageIndex(rows, 'a2')).toBe(3)

    const next = [{ id: 'u0', role: 'user' }, ...rows]

    expect(threadMessageIndex(next, 'a2')).toBe(4)
    expect(threadUserOrdinal(next, 'u2')).toBe(2)
  })
})
