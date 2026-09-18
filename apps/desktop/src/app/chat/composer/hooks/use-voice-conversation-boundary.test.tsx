import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { type ChatMessage, collectUnspokenTurnSpeech } from '@/lib/chat-messages'
import { stopVoicePlayback } from '@/lib/voice-playback'

import { useVoiceConversation } from './use-voice-conversation'

const mocks = vi.hoisted(() => ({
  config: vi.fn(),
  mic: {
    cancel: vi.fn(),
    start: vi.fn(async () => undefined),
    stop: vi.fn(async () => ({
      audio: new Blob(['fixture']),
      heardSpeech: true,
      durationMs: 900
    }))
  }
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  hermesApi: mocks.config,
  speakText: vi.fn()
}))
vi.mock('@/api/client', () => ({ profileScoped: (value: unknown) => value }))
vi.mock('./use-mic-recorder', () => ({ useMicRecorder: () => ({ handle: mocks.mic, level: 0 }) }))
vi.mock('@/lib/voice-barge-in', () => ({ monitorSpeechDuringPlayback: () => vi.fn() }))
vi.mock('@/lib/thinking-sound', () => ({ startThinkingSound: vi.fn(), stopThinkingSound: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { notifications: { voice: {} } } }) }))

class TestAudio extends EventTarget {
  static instances: TestAudio[] = []
  src: string
  constructor(src: string) {
    super()
    this.src = src
    TestAudio.instances.push(this)
  }
  play = vi.fn(async () => undefined)
  pause = vi.fn()
  load = vi.fn()
}

afterEach(() => {
  cleanup()
  stopVoicePlayback()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('speaks a sealed narration while busy and keeps the session open for the final reply', async () => {
  vi.useFakeTimers()
  TestAudio.instances = []
  vi.stubGlobal('Audio', TestAudio)
  vi.stubGlobal(
    'URL',
    class extends URL {
      static createObjectURL = vi.fn(() => 'blob:fixture')
      static revokeObjectURL = vi.fn()
    }
  )
  mocks.config.mockResolvedValue({
    ok: true,
    stt: { mode: 'relay' },
    tts: {
      mode: 'direct',
      wire: 'openai-speech',
      provider: 'openai',
      base_url: 'https://tts.invalid/v1',
      api_key: 'fixture-only',
      model: 'tts-fixture',
      voice: 'fixture',
      speed: null
    }
  })
  const inputs: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, options: RequestInit) => {
      inputs.push(JSON.parse(options.body as string).input)

      return { ok: true, arrayBuffer: async () => new Uint8Array([1]).buffer }
    })
  )
  const narration = 'Let me check the live state of the branch.'
  const answer = 'The branch is clean and the check is complete.'
  const messages: ChatMessage[] = []

  const hook = renderHook(
    ({ busy }) =>
      useVoiceConversation({
        busy,
        enabled: true,
        consumePendingResponse: vi.fn(),
        onSubmit: vi.fn(async () => {
          hook.rerender({ busy: true })
        }),
        onTranscribeAudio: async () => 'Check the branch',
        pendingResponse: () => collectUnspokenTurnSpeech(messages, null)
      }),
    { initialProps: { busy: false } }
  )

  await act(async () => {
    await hook.result.current.start()
  })
  await act(async () => {
    hook.result.current.stopTurn()
  })
  messages.push({ id: 'narration', role: 'assistant', pending: true, parts: [{ type: 'text', text: narration }] })
  hook.rerender({ busy: true })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(inputs).toEqual([])
  messages[0].pending = false
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(inputs).toEqual([narration])
  expect(TestAudio.instances[0].play).toHaveBeenCalledOnce()
  await act(async () => {
    TestAudio.instances[0].dispatchEvent(new Event('ended'))
  })
  expect(mocks.mic.start).toHaveBeenCalledTimes(1)
  messages.push({ id: 'answer', role: 'assistant', pending: false, parts: [{ type: 'text', text: answer }] })
  hook.rerender({ busy: false })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })
  expect(inputs).toEqual([narration, answer])
})
