import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'

const broadcastTranscriptChanged = vi.hoisted(() => vi.fn())

vi.mock('@/store/transcript-sync', () => ({
  broadcastTranscriptChanged
}))

const SID = 'rt-peer'
const STORED = 'stored-peer'

describe('completed turn notifies peer windows (#65047)', () => {
  afterEach(() => {
    cleanup()
    broadcastTranscriptChanged.mockClear()
  })

  it('broadcasts the stored session when a turn completes', () => {
    const states = new Map([[SID, createClientSessionState(STORED)]])
    const stream = renderMessageStream(SID, { states })

    act(() => {
      stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' })
    })

    expect(broadcastTranscriptChanged).toHaveBeenCalledWith(expect.objectContaining({ sessionId: STORED }))
  })

  it('does not broadcast a completed runtime that has no stored session', () => {
    const stream = renderMessageStream(SID)

    act(() => {
      stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' })
    })

    expect(broadcastTranscriptChanged).not.toHaveBeenCalled()
  })
})
