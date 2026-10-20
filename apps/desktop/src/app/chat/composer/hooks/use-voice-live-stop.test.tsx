import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { VoiceLiveHandlers } from '@/lib/voice-live'
import { applyVoiceStopPhraseFromConfig } from '@/store/voice-prefs'

import { useVoiceLiveConversation } from './use-voice-live-conversation'

const sessions: VoiceLiveHandlers[] = []
const close = vi.fn()

vi.mock('@/lib/voice-live', () => ({
  VoiceLiveSession: class {
    constructor(handlers: VoiceLiveHandlers) {
      sessions.push(handlers)
    }

    start = vi.fn(async () => undefined)
    close = close
    speak = vi.fn()
    setMuted = vi.fn()
  }
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  applyVoiceStopPhraseFromConfig(null)
  sessions.length = 0
  close.mockClear()
})

it('checks settled live transcripts against the current config without swallowing a longer request', async () => {
  vi.useFakeTimers()
  const onStopWord = vi.fn()

  const hook = renderHook(() =>
    useVoiceLiveConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onStopWord,
      onSubmit: vi.fn(),
      pendingResponse: () => null,
      seedHistory: () => []
    })
  )

  await act(() => hook.result.current.start())
  const handlers = sessions.at(-1)!

  for (const text of ['stop', '그만하고 다음 작업']) {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: text === 'stop' ? [] : ['그만'] } })
    await act(async () => {
      handlers.onTranscript?.({ speaker: 'user', text, startMs: 0, endMs: 100 })
      await vi.advanceTimersByTimeAsync(1_600)
    })
    expect(onStopWord).not.toHaveBeenCalled()
  }

  await act(async () => {
    handlers.onTranscript?.({ speaker: 'user', text: '그만!'.normalize('NFD'), startMs: 0, endMs: 100 })
    await vi.advanceTimersByTimeAsync(1_600)
  })
  expect(onStopWord).toHaveBeenCalledTimes(1)
  expect(close).toHaveBeenCalledTimes(1)
})

it('uses stop configuration for live delegations before submitting a Hermes turn', async () => {
  for (const { phrases, text, stops } of [
    { phrases: ['대화 종료'], text: '대화 종료!', stops: true },
    { phrases: [], text: 'stop', stops: false },
    { phrases: ['stop'], text: 'stop the docker container', stops: false }
  ]) {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: phrases } })
    const onStopWord = vi.fn()
    const onSubmit = vi.fn()

    const hook = renderHook(() =>
      useVoiceLiveConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onStopWord,
        onSubmit,
        pendingResponse: () => null,
        seedHistory: () => []
      })
    )

    await act(() => hook.result.current.start())
    await act(async () => {
      sessions.at(-1)!.onDelegation('delegation', [{ speaker: 'user', text, startMs: 0, endMs: 100 }])
    })
    expect(onStopWord).toHaveBeenCalledTimes(stops ? 1 : 0)
    expect(onSubmit).toHaveBeenCalledTimes(stops ? 0 : 1)
    await act(() => hook.result.current.end())
    hook.unmount()
  }
})
