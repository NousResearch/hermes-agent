import { describe, expect, it } from 'vitest'

import { userTurnFromEnd } from './slash-session-commands'

// After a compression the owner's resume snapshot also carries the turns it summarized away, marked
// `archived`: history to show, never an /undo or /retry target (the owner refuses a rewind to one).
const rows = [
  { archived: true, content: 'old question', role: 'user', row_id: 1 },
  { archived: true, content: 'old answer', role: 'assistant', row_id: 2 },
  { content: '[CONTEXT COMPACTION] earlier turns', role: 'user', row_id: 3 },
  { content: 'live question', role: 'user', row_id: 4 },
  { content: 'live answer', role: 'assistant', row_id: 5 }
]

describe('userTurnFromEnd', () => {
  it('chooses only among live user turns', () => {
    expect(userTurnFromEnd(rows, 1)?.row_id).toBe(4)
    expect(userTurnFromEnd(rows, 2)).toBeNull()
  })

  it('finds nothing to rewind once the live turns are gone', () => {
    expect(userTurnFromEnd(rows.slice(0, 3), 1)).toBeNull()
  })
})
