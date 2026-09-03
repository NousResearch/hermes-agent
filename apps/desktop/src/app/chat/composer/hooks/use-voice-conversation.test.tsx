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
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

let realtimeCommitted: ((text: string) => void) | null = null

const realtimeSession = {
  close: vi.fn(),
  commit: vi.fn(() => realtimeCommitted?.('kick off the task')),
  mute: vi.fn(),
  unmute: vi.fn()
}

const startRealtimeScribe = vi.fn<
  (options: { onCommitted: (text: string) => void }) => Promise<typeof realtimeSession | null>
>(async ({ onCommitted }) => {
  realtimeCommitted = onCommitted

  return realtimeSession
})

vi.mock('@/lib/realtime-scribe', () => ({ startRealtimeScribe: (options: { onCommitted: (text: string) => void }) => startRealtimeScribe(options) }))

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

function renderConversation(overrides: { onInterrupt?: () => void; transcript?: string } = {}) {
  const onInterrupt = overrides.onInterrupt ?? vi.fn()

  // Mirrors the real app: submitting a turn makes the agent busy.
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const onSubmit = vi.fn(async () => {
    onBusyChange.current(true)
  })

  const onStopWord = vi.fn()

  // Realtime supplies the kickoff transcript directly; record/upload is used
  // either as the compatibility path or for a captured barge utterance.
  const onTranscribeAudio = vi.fn(async () => overrides.transcript ?? 'kick off the task')

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
    realtimeCommitted = null
    vi.clearAllMocks()
    startRealtimeScribe.mockImplementation(async ({ onCommitted }: { onCommitted: (text: string) => void }) => {
      realtimeCommitted = onCommitted

      return realtimeSession
    })
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('uses a persistent realtime transcript instead of record-upload when available', async () => {
    const { hook, onSubmit, onTranscribeAudio } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(startRealtimeScribe).toHaveBeenCalledTimes(1))
    expect(realtimeSession.unmute).toHaveBeenCalled()

    await act(async () => {
      realtimeCommitted?.('hello in realtime')
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('hello in realtime'))
    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(realtimeSession.mute).toHaveBeenCalled()

    await act(async () => {
      await hook.result.current.end()
    })
    expect(realtimeSession.close).toHaveBeenCalled()
  })

  it('closes a realtime session that resolves after the user mutes', async () => {
    let resolveRealtime: ((session: typeof realtimeSession) => void) | null = null
    startRealtimeScribe.mockImplementationOnce(
      ({ onCommitted }: { onCommitted: (text: string) => void }) => {
        realtimeCommitted = onCommitted

        return new Promise(resolve => {
          resolveRealtime = resolve
        })
      }
    )
    const { hook } = renderConversation()

    let starting: Promise<void>
    act(() => {
      starting = hook.result.current.start()
    })
    await waitFor(() => expect(startRealtimeScribe).toHaveBeenCalledTimes(1))

    act(() => {
      hook.result.current.toggleMute()
    })
    expect(hook.result.current.muted).toBe(true)

    await act(async () => {
      resolveRealtime?.(realtimeSession)
      await starting
    })

    expect(realtimeSession.close).toHaveBeenCalled()
    expect(realtimeSession.unmute).not.toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })

  it('falls back to record-upload when realtime Scribe is unavailable', async () => {
    startRealtimeScribe.mockResolvedValueOnce(null)
    const { hook, onSubmit, onTranscribeAudio } = renderConversation()

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

    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalled())
    expect(onSubmit).toHaveBeenCalledWith('kick off the task')
  })

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
