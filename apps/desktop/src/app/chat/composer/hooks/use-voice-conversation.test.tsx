import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// The full-duplex contract: the barge monitor is live across the WHOLE agent
// turn — generation (thinking) and playback (speaking) — so speaking over the
// model interrupts it mid-generation instead of the mic being deaf until TTS
// starts (the Windows report: interruption "never works" because the deaf
// window covered generation, and playback bleed made the old monitor's
// trigger unreachable).

const monitorCalls: BargeMonitorCallbacks[] = []
const stopMonitor = vi.fn()

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: (callbacks: BargeMonitorCallbacks) => {
    monitorCalls.push(callbacks)

    return stopMonitor
  }
}))

const markVoicePlaybackInterrupted = vi.fn()
const stopVoicePlayback = vi.fn()

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: () => markVoicePlaybackInterrupted(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: () => stopVoicePlayback()
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => true),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

interface HookProps {
  busy: boolean
}

function renderConversation(
  overrides: {
    onInterrupt?: () => void
    onTranscribeAudio?: (audio: Blob) => Promise<string>
    transcript?: string
  } = {}
) {
  const onInterrupt = overrides.onInterrupt ?? vi.fn()

  // Mirrors the real app: submitting a turn makes the agent busy.
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const onSubmit = vi.fn(async () => {
    onBusyChange.current(true)
  })

  const onStopWord = vi.fn()

  // First transcription is the turn that starts the conversation; subsequent
  // ones are barge captures (the overridable transcript).
  let transcriptions = 0

  const onTranscribeAudio =
    overrides.onTranscribeAudio ??
    vi.fn(async () =>
      transcriptions++ === 0 ? 'kick off the task' : (overrides.transcript ?? 'and another thing')
    )

  const hook = renderHook(
    ({ busy }: HookProps) =>
      useVoiceConversation({
        busy,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onInterrupt,
        onStopWord,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      }),
    { initialProps: { busy: false } }
  )

  onBusyChange.current = busy => hook.rerender({ busy })

  return { hook, onInterrupt, onStopWord, onSubmit, onTranscribeAudio }
}

/** Drive the hook into the generation phase (turn submitted, model working). */
async function enterThinking(hook: ReturnType<typeof renderConversation>['hook']) {
  await act(async () => {
    await hook.result.current.start()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('listening'))

  micHandle.stop.mockResolvedValueOnce({
    audio: new Blob(['q'], { type: 'audio/webm' }),
    durationMs: 900,
    heardSpeech: true
  })

  await act(async () => {
    hook.result.current.stopTurn()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
}

describe('useVoiceConversation full-duplex barge-in', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(true)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('arms the barge monitor during generation (before any reply audio exists)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)

    await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    // busy=true + thinking → the full-duplex monitor must be live.
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
  })

  it('does not enter listening or arm a timeout when mic start is cancelled', async () => {
    micHandle.start.mockResolvedValueOnce(false)
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })

    expect(hook.result.current.status).toBe('idle')
    expect(micHandle.stop).not.toHaveBeenCalled()
  })

  it('does not submit a normal turn when the conversation ends during transcription', async () => {
    let resolveTranscription!: (transcript: string) => void

    const transcription = new Promise<string>(resolve => {
      resolveTranscription = resolve
    })

    const { hook, onSubmit } = renderConversation({
      onTranscribeAudio: vi.fn(async () => transcription)
    })

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['q'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })

    act(() => {
      hook.result.current.stopTurn()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('transcribing'))

    await act(async () => {
      await hook.result.current.end()
      resolveTranscription('stale normal turn')
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })

  it('does not submit a barge capture when the conversation ends during transcription', async () => {
    let calls = 0
    let resolveTranscription!: (transcript: string) => void

    const transcription = new Promise<string>(resolve => {
      resolveTranscription = resolve
    })

    const { hook, onSubmit } = renderConversation({
      onTranscribeAudio: vi.fn(async () => {
        calls += 1

        return calls === 1 ? 'kick off the task' : transcription
      })
    })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
    const monitor = monitorCalls.at(-1)
    act(() => monitor?.onSpeech())
    hook.rerender({ busy: false })

    act(() => {
      monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
    })
    await waitFor(() => expect(hook.result.current.status).toBe('transcribing'))

    await act(async () => {
      await hook.result.current.end()
      resolveTranscription('stale barge turn')
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('stale barge turn')
    expect(hook.result.current.status).toBe('idle')
  })

  it('does not submit a barge capture when the conversation ends during interrupt settling', async () => {
    const { hook, onSubmit } = renderConversation({ transcript: 'stale after wait' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
    const monitor = monitorCalls.at(-1)
    act(() => monitor?.onSpeech())

    vi.useFakeTimers()

    try {
      const submission = monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
      await Promise.resolve()
      await act(async () => {
        await hook.result.current.end()
        vi.advanceTimersByTime(100)
        await submission
      })

      expect(onSubmit).not.toHaveBeenCalledWith('stale after wait')
      expect(hook.result.current.status).toBe('idle')
    } finally {
      vi.useRealTimers()
    }
  })

  it('allows a fresh generation to submit after a cancelled turn', async () => {
    let calls = 0
    let resolveTranscription!: (transcript: string) => void

    const staleTranscription = new Promise<string>(resolve => {
      resolveTranscription = resolve
    })

    const { hook, onSubmit } = renderConversation({
      onTranscribeAudio: vi.fn(async () => {
        calls += 1

        if (calls === 1) {
          return staleTranscription
        }

        return 'fresh turn'
      })
    })

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['q'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })
    act(() => hook.result.current.stopTurn())
    await waitFor(() => expect(hook.result.current.status).toBe('transcribing'))
    await act(async () => {
      await hook.result.current.end()
      resolveTranscription('stale turn')
    })

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['fresh'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })
    await act(async () => hook.result.current.stopTurn())
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('fresh turn'))
  })

  it('interrupts the in-flight turn when speech trips mid-generation', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(markVoicePlaybackInterrupted).toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('submits the captured interruption once the interrupt settles (busy clears)', async () => {
    const { hook, onSubmit } = renderConversation({ transcript: 'no, do it differently' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })

    // Interrupt lands → the turn ends → busy flips false.
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('no, do it differently'))
  })

  it('does not interrupt when speech trips during playback (turn already done)', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    // Turn finished; playback phase.
    hook.rerender({ busy: false })

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).not.toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('a spoken stop command in the barge capture ends the conversation instead of submitting', async () => {
    const { hook, onStopWord, onSubmit } = renderConversation({ transcript: 'stop' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['s'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onStopWord).toHaveBeenCalledTimes(1))
    // Only the kickoff turn was submitted — the "stop" capture never was.
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('stop')
  })

  it('re-arms a single monitor per turn (idempotent ensure)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const armed = monitorCalls.length

    // Effect re-runs (busy toggles, status changes) must not open more mics.
    hook.rerender({ busy: true })
    hook.rerender({ busy: true })

    expect(monitorCalls.length).toBe(armed)
  })
})
