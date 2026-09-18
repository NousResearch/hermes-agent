import { beforeEach, describe, expect, it } from 'vitest'

import { $transcriptTailBySessionId, clearTranscriptTailPaging, recordTranscriptTail } from './transcript-tail'

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

  it('keeps a no-op re-recorded entry at the MRU end so it survives the next eviction', () => {
    clearTranscriptTailPaging()
    recordTranscriptTail('active', page(5))

    for (let i = 0; i < 255; i += 1) {
      recordTranscriptTail(`other-${i}`, page(5))
    }

    // Identical re-record: no publish, but it must still count as recent use.
    recordTranscriptTail('active', page(5))
    recordTranscriptTail('newcomer', page(5))

    expect($transcriptTailBySessionId.get().active).toBeDefined()
    expect($transcriptTailBySessionId.get()['other-0']).toBeUndefined()
  })
})
