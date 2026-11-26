import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { retainVoiceStreamPaint } from '@/store/voice-stream-paint'

import { renderMessageStream } from './test-harness'

const SID = 'hidden-voice-flush-session'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it.each([true, false])('keeps hidden stream batching responsive only with a live voice consumer: %s', async active => {
  vi.useFakeTimers()
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('hidden')
  const stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
  act(() => {
    stream.handleEvent({ payload: { running: true }, session_id: SID, type: 'session.info' })
    stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' })
    stream.handleEvent({ payload: { text: 'first ' }, session_id: SID, type: 'message.delta' })
  })
  await act(async () => vi.advanceTimersByTimeAsync(300))
  expect(stream.text()).toBe('first ')

  // GPT Live owns transcript paint without STT/TTS leases or read-aloud playback.
  const release = active ? retainVoiceStreamPaint() : undefined

  try {
    act(() => stream.handleEvent({ payload: { text: 'second' }, session_id: SID, type: 'message.delta' }))
    await act(async () => vi.advanceTimersByTimeAsync(60))
    expect(stream.text()).toBe(active ? 'first second' : 'first ')
    await act(async () => vi.advanceTimersByTimeAsync(250))
    expect(stream.text()).toBe('first second')
  } finally {
    release?.()
  }
})
