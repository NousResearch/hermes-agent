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
  start: vi.fn<() => Promise<void>>(async () => undefined),
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
  enabled?: boolean
}

function renderConversation(
  overrides: { onInterrupt?: () => void; transcript?: string; transcribe?: (audio: Blob) => Promise<string> } = {}
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

  const onTranscribeAudio = vi.fn(
    overrides.transcribe ??
      (async () => (transcriptions++ === 0 ? 'kick off the task' : (overrides.transcript ?? 'and another thing')))
  )

  const hook = renderHook(
    ({ busy, enabled = true }: HookProps) =>
      useVoiceConversation({
        busy,
        consumePendingResponse: vi.fn(),
        enabled,
        onInterrupt,
        onStopWord,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      }),
    { initialProps: { busy: false } as HookProps }
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
    micHandle.start.mockResolvedValue(undefined)
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
    await act(async () => {
      hook.rerender({ busy: false })
    })

    await act(async () => {
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

describe('useVoiceConversation lifecycle ownership', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it.each(['end', 'disable', 'unmount', 'mute'] as const)(
    'drops pending transcription after %s without submitting or handling a stop command',
    async cancellation => {
      let resolveTranscript!: (text: string) => void

      const { hook, onSubmit, onStopWord, onTranscribeAudio } = renderConversation({
        transcribe: () =>
          new Promise(resolve => {
            resolveTranscript = resolve
          })
      })

      await act(async () => {
        await hook.result.current.start()
      })
      micHandle.stop.mockResolvedValueOnce({
        audio: new Blob(['q'], { type: 'audio/webm' }),
        durationMs: 900,
        heardSpeech: true
      })
      await act(async () => {
        hook.result.current.stopTurn()
      })
      expect(onTranscribeAudio).toHaveBeenCalledOnce()

      await act(async () => {
        if (cancellation === 'end') {
          await hook.result.current.end()
        }

        if (cancellation === 'disable') {
          hook.rerender({ busy: false, enabled: false })
        }

        if (cancellation === 'unmount') {
          hook.unmount()
        }

        if (cancellation === 'mute') {
          hook.result.current.toggleMute()
        }
      })
      await act(async () => {
        resolveTranscript('do not send this after cancellation')
      })

      expect(onSubmit).not.toHaveBeenCalled()
      expect(onStopWord).not.toHaveBeenCalled()
      expect(micHandle.start).toHaveBeenCalledTimes(1)
    }
  )

  it.each(['end', 'unmount', 'mute'] as const)(
    'releases the generation microphone on %s and ignores a queued barge callback',
    async cancellation => {
      const { hook, onInterrupt, onSubmit, onTranscribeAudio } = renderConversation()
      await enterThinking(hook)
      expect(monitorCalls).toHaveLength(1)
      const monitor = monitorCalls[0]
      const stoppedBefore = stopMonitor.mock.calls.length
      await act(async () => {
        if (cancellation === 'end') {
          await hook.result.current.end()
        }

        if (cancellation === 'unmount') {
          hook.unmount()
        }

        if (cancellation === 'mute') {
          hook.result.current.toggleMute()
        }
      })

      expect(stopMonitor.mock.calls.length).toBeGreaterThan(stoppedBefore)
      await act(async () => {
        monitor.onSpeech()
        monitor.onUtterance?.(new Blob(['late'], { type: 'audio/webm' }))
      })
      expect(onInterrupt).not.toHaveBeenCalled()
      expect(onTranscribeAudio).toHaveBeenCalledTimes(1)
      expect(onSubmit).toHaveBeenCalledTimes(1)
    }
  )

  it('does not resurrect listening after a pending microphone start is cancelled', async () => {
    let resolveStart!: () => void
    micHandle.start.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveStart = resolve
        })
    )
    const { hook } = renderConversation()
    let start!: Promise<void>
    act(() => {
      start = hook.result.current.start()
    })
    await waitFor(() => expect(micHandle.start).toHaveBeenCalledOnce())
    await act(async () => {
      await hook.result.current.end()
    })
    await act(async () => {
      resolveStart()
      await start
    })
    expect(hook.result.current.status).toBe('idle')
  })
})
