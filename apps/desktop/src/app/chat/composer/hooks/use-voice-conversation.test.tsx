// @vitest-environment jsdom
import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  handle: {
    cancel: vi.fn(),
    start: vi.fn(async (_options?: unknown) => undefined),
    stop: vi.fn(async () => ({ audio: new Blob(['audio']), heardSpeech: true }))
  },
  getHermesConfigRecord: vi.fn(async () => ({})),
  notify: vi.fn(),
  notifyError: vi.fn(),
  playSpeechText: vi.fn(async () => true),
  realtimeCallbacks: null as Record<string, (...args: any[]) => void> | null,
  realtimeCancelInput: vi.fn(),
  realtimeConnect: vi.fn(async () => undefined),
  realtimeDisconnect: vi.fn(),
  realtimeSetMuted: vi.fn(),
  saveHermesConfig: vi.fn(async (_config?: unknown): Promise<void> => undefined),
  stopVoicePlayback: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'microphone failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: mocks.getHermesConfigRecord,
  saveHermesConfig: mocks.saveHermesConfig
}))

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: mocks.playSpeechText,
  stopVoicePlayback: mocks.stopVoicePlayback
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: {
      cancel: mocks.handle.cancel,
      start: mocks.handle.start,
      stop: mocks.handle.stop
    },
    level: 0.25
  })
}))

vi.mock('@/lib/realtime-voice-session', () => ({
  RealtimeVoiceSession: class {
    constructor(callbacks: Record<string, (...args: any[]) => void>) {
      mocks.realtimeCallbacks = callbacks
    }

    cancelInput = mocks.realtimeCancelInput
    connect = mocks.realtimeConnect
    disconnect = mocks.realtimeDisconnect
    setMuted = mocks.realtimeSetMuted
  }
}))

import { $voiceInputMode, setVoiceInputMode } from '@/store/voice-prefs'

import { useVoiceConversation } from './use-voice-conversation'

interface HookProps {
  busy: boolean
  enabled: boolean
  mode: 'legacy' | 'realtime'
}

const mountedRoots: Root[] = []

async function waitFor(assertion: () => void) {
  let lastError: unknown

  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      assertion()

      return
    } catch (error) {
      lastError = error
      await act(async () => new Promise(resolve => window.setTimeout(resolve, 0)))
    }
  }

  throw lastError
}

function renderVoiceHook(
  initialProps: HookProps,
  options: Omit<Parameters<typeof useVoiceConversation>[0], keyof HookProps>
) {
  let current: ReturnType<typeof useVoiceConversation>
  let props = initialProps
  const container = document.createElement('div')
  const root = createRoot(container)
  mountedRoots.push(root)

  function Harness(): ReactNode {
    current = useVoiceConversation({ ...options, ...props })

    return null
  }

  act(() => root.render(<Harness />))

  return {
    rerender(next: HookProps) {
      props = next
      act(() => root.render(<Harness />))
    },
    result: {
      get current() {
        return current
      }
    }
  }
}

function setup(initial: HookProps = { busy: false, enabled: false, mode: 'legacy' }) {
  const onSubmit = vi.fn(async () => undefined)
  const onTranscribeAudio = vi.fn(async () => 'legacy transcript')
  const consumePendingResponse = vi.fn()
  const pendingResponse = vi.fn(() => null)

  const hook = renderVoiceHook(initial, {
    consumePendingResponse,
    onSubmit,
    onTranscribeAudio,
    pendingResponse,
    sessionId: 'hermes-session-1'
  })

  return { consumePendingResponse, hook, onSubmit, onTranscribeAudio, pendingResponse }
}

describe('useVoiceConversation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.realtimeCallbacks = null
    mocks.handle.stop.mockResolvedValue({ audio: new Blob(['audio']), heardSpeech: true })
    mocks.getHermesConfigRecord.mockResolvedValue({})
    $voiceInputMode.set('legacy')
  })

  afterEach(() => {
    for (const root of mountedRoots.splice(0)) {
      act(() => root.unmount())
    }
  })

  it('keeps the Legacy recorder -> transcribe -> existing submit path unchanged', async () => {
    const { hook, onSubmit, onTranscribeAudio } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'legacy' })
    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(1))

    const options = mocks.handle.start.mock.calls[0]![0] as { onSilence: () => void }

    await act(async () => {
      options.onSilence()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(onTranscribeAudio).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('legacy transcript')
    expect(mocks.realtimeConnect).not.toHaveBeenCalled()
  })

  it('uses one Hermes submit/tool pipeline per Realtime transcript and makes no Legacy network call', async () => {
    const { hook, onSubmit, onTranscribeAudio } = setup()
    const observedToolEvents: Array<{ name: string; type: string }> = []
    onSubmit.mockImplementationOnce(async () => {
      observedToolEvents.push({ name: 'read_file', type: 'tool.start' })
    })

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeConnect).toHaveBeenCalledWith({ sessionId: 'hermes-session-1' }))

    expect(mocks.handle.start).not.toHaveBeenCalled()
    expect(onTranscribeAudio).not.toHaveBeenCalled()

    await act(async () => {
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-1', text: '첫 번째 질문' })
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-1', text: '첫 번째 질문' })
      await Promise.resolve()
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('첫 번째 질문')
    expect(observedToolEvents).toEqual([{ name: 'read_file', type: 'tool.start' }])
    expect(mocks.realtimeCancelInput).not.toHaveBeenCalled()
  })

  it('queues barge-in while Hermes is busy, stops only TTS, then submits once when ready', async () => {
    const { hook, onSubmit } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeCallbacks).not.toBeNull())
    hook.rerender({ busy: true, enabled: true, mode: 'realtime' })
    expect(mocks.realtimeDisconnect).not.toHaveBeenCalled()

    act(() => {
      mocks.realtimeCallbacks?.onSpeechStarted()
      mocks.realtimeCallbacks?.onTranscript({ id: 'item-barge', text: '새 질문' })
    })

    expect(mocks.stopVoicePlayback).toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit).toHaveBeenCalledWith('새 질문')
  })

  it('surfaces connection errors for explicit retry and cleans the peer on end', async () => {
    const { hook } = setup()

    hook.rerender({ busy: false, enabled: true, mode: 'realtime' })
    await waitFor(() => expect(mocks.realtimeCallbacks).not.toBeNull())

    act(() => mocks.realtimeCallbacks?.onError(new Error('offline')))
    expect(hook.result.current.status).toBe('error')

    await act(async () => hook.result.current.start())
    expect(mocks.realtimeConnect).toHaveBeenCalledTimes(2)

    await act(async () => hook.result.current.end())
    expect(mocks.realtimeDisconnect).toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })
})

describe('Realtime voice preference', () => {
  it('publishes Realtime mode only after the backend feature gate is saved', async () => {
    let releaseSave: (() => void) | undefined

    mocks.getHermesConfigRecord.mockResolvedValue({ voice: { auto_tts: true } })
    mocks.saveHermesConfig.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          releaseSave = resolve
        })
    )

    const saving = setVoiceInputMode('realtime')

    await waitFor(() => expect(mocks.saveHermesConfig).toHaveBeenCalledTimes(1))
    expect($voiceInputMode.get()).toBe('legacy')
    expect(mocks.saveHermesConfig).toHaveBeenCalledWith({
      voice: {
        auto_tts: true,
        input_mode: 'realtime',
        realtime: { enabled: true }
      }
    })

    releaseSave?.()
    await saving
    expect($voiceInputMode.get()).toBe('realtime')
  })
})
