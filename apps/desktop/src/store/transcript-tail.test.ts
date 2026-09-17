import { beforeEach, describe, expect, it } from 'vitest'

import { $transcriptTailBySessionId, recordTranscriptTail } from './transcript-tail'

const page = (count: number, limit = 10) => ({
  messages: Array.from({ length: count }, (_, i) => ({ id: `m${i}` })),
  pagination: { limit, offset: 0 }
}) as never

describe('recordTranscriptTail no-op suppression', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  it('does not notify on an identical re-record (#113842)', () => {
    recordTranscriptTail('s1', page(5))

    let notifications = 0
    const unsub = $transcriptTailBySessionId.subscribe(() => {
      notifications += 1
    })
    notifications = 0

    recordTranscriptTail('s1', page(5))
    unsub()

    expect(notifications).toBe(0)
  })

  it('notifies when the tail actually advances', () => {
    recordTranscriptTail('s1', page(5))

    let notifications = 0
    const unsub = $transcriptTailBySessionId.subscribe(() => {
      notifications += 1
    })
    notifications = 0

    recordTranscriptTail('s1', page(7))
    unsub()

    expect(notifications).toBe(1)
  })
})
