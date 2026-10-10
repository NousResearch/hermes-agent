import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'stream-session'

let stream: MessageStreamHarness

describe('stream delta delivery', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('delivers a delta while animation frames are parked', async () => {
    // An Electron renderer the compositor has parked: rAF accepts the callback
    // but never runs it. A frame-gated flush strands the answer here until some
    // later focus/input event wakes a frame.
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    vi.useFakeTimers()

    stream = renderMessageStream(SID)
    await act(async () => {
      await Promise.resolve()
    })

    // Reach the frame-gated branch: it is only taken once the coalescing floor
    // has already elapsed since the previous flush. Send one delta, let it
    // flush, then idle past the floor so the NEXT delta schedules immediately.
    act(() => stream.handleEvent({ payload: { text: 'first ' }, session_id: SID, type: 'message.delta' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS * 2)
    })

    act(() => stream.handleEvent({ payload: { text: 'and the rest' }, session_id: SID, type: 'message.delta' }))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    expect(stream.state()?.messages.at(-1)?.parts).toMatchObject([{ type: 'text', text: 'first and the rest' }])
    // The flush must not have depended on a frame: this mock parks every rAF
    // callback, yet the text arrived. runFlush still registers its
    // adaptive-floor measurement callback here; that one is allowed to wait
    // for a frame that may never come.
    expect(rafSpy).toHaveBeenCalled()
  })
})
