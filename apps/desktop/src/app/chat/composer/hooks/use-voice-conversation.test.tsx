import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'

import type { MicRecorderOptions, MicRecording } from './use-mic-recorder'
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
  start: vi.fn<(options?: MicRecorderOptions) => Promise<void>>(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  // The real hook returns a fresh handle object on every render.
  useMicRecorder: () => ({ handle: { ...micHandle }, level: 0, recording: false })
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

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(next => {
    resolve = next
  })

  return { promise, resolve }
}

function renderConversation(overrides: { onInterrupt?: () => void; transcript?: string } = {}) {
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

  const onTranscribeAudio = vi.fn(async () =>
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
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it.each(['stale-first', 'replacement-first'] as const)(
    'keeps a stale cancelled start inert when unmuting starts its replacement (%s)',
    async resolveOrder => {
      const staleStart = deferred<void>()
      const replacementStart = deferred<void>()
      const startSignals: AbortSignal[] = []

      micHandle.start
        .mockImplementationOnce(options => {
          startSignals.push(options?.signal as AbortSignal)

          return staleStart.promise
        })
        .mockImplementationOnce(options => {
          startSignals.push(options?.signal as AbortSignal)

          return replacementStart.promise
        })

      const hook = renderHook(() =>
        useVoiceConversation({
          busy: false,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          onTranscribeAudio: vi.fn(async () => ''),
          pendingResponse: () => null
        })
      )

      let starting!: Promise<void>

      act(() => {
        starting = hook.result.current.start()
      })
      await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(1))

      act(() => hook.result.current.toggleMute())
      expect(hook.result.current).toMatchObject({ muted: true, status: 'idle' })
      expect(micHandle.cancel).toHaveBeenCalled()

      act(() => hook.result.current.toggleMute())
      await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(2))
      expect(startSignals[0]?.aborted).toBe(true)
      expect(startSignals[1]?.aborted).toBe(false)

      await act(async () => {
        if (resolveOrder === 'stale-first') {
          staleStart.resolve()
          await starting
          expect(hook.result.current).toMatchObject({ muted: false, status: 'idle' })
          replacementStart.resolve()
          await replacementStart.promise
        } else {
          replacementStart.resolve()
          await replacementStart.promise
          staleStart.resolve()
          await starting
        }
      })

      expect(startSignals[1]?.aborted).toBe(false)
      await waitFor(() => expect(hook.result.current).toMatchObject({ muted: false, status: 'listening' }))
    }
  )

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

  it('ends the conversation after the configured interval without user speech', async () => {
    vi.useFakeTimers()
    const onIdleTimeout = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 1_000,
        onIdleTimeout,
        onSubmit: vi.fn(),
        onTranscribeAudio: vi.fn(async () => ''),
        pendingResponse: () => null
      })
    )

    try {
      await act(async () => {
        await hook.result.current.start()
      })
      expect(hook.result.current.status).toBe('listening')

      await act(async () => {
        vi.advanceTimersByTime(999)
      })
      expect(onIdleTimeout).not.toHaveBeenCalled()

      await act(async () => {
        vi.advanceTimersByTime(1)
      })
      expect(onIdleTimeout).toHaveBeenCalledTimes(1)
      expect(micHandle.cancel).toHaveBeenCalled()
      expect(hook.result.current.status).toBe('idle')
    } finally {
      vi.useRealTimers()
    }
  })

  it('restarts the inactivity interval after transcribed user speech', async () => {
    vi.useFakeTimers()
    const onIdleTimeout = vi.fn()
    const onSubmit = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 1_000,
        onIdleTimeout,
        onSubmit,
        onTranscribeAudio: vi.fn(async () => 'still here'),
        pendingResponse: () => null
      })
    )

    try {
      await act(async () => {
        await hook.result.current.start()
        vi.advanceTimersByTime(600)
      })
      micHandle.stop.mockResolvedValueOnce({
        audio: new Blob(['speech'], { type: 'audio/webm' }),
        durationMs: 400,
        heardSpeech: true
      })

      await act(async () => {
        hook.result.current.stopTurn()
        await Promise.resolve()
      })
      expect(onSubmit).toHaveBeenCalledWith('still here')

      await act(async () => {
        vi.advanceTimersByTime(400)
      })
      expect(onIdleTimeout).not.toHaveBeenCalled()

      await act(async () => {
        vi.advanceTimersByTime(600)
      })
      expect(onIdleTimeout).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('clears the inactivity interval on unmount without later callbacks or resource actions', async () => {
    vi.useFakeTimers()
    const onIdleTimeout = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 1_000,
        onIdleTimeout,
        onSubmit: vi.fn(),
        onTranscribeAudio: vi.fn(async () => ''),
        pendingResponse: () => null
      })
    )

    try {
      await act(async () => {
        await hook.result.current.start()
      })
      hook.unmount()
      const cancelCallsAfterUnmount = micHandle.cancel.mock.calls.length
      const playbackStopsAfterUnmount = stopVoicePlayback.mock.calls.length

      await act(async () => {
        vi.advanceTimersByTime(1_000)
      })

      expect(onIdleTimeout).not.toHaveBeenCalled()
      expect(micHandle.cancel).toHaveBeenCalledTimes(cancelCallsAfterUnmount)
      expect(stopVoicePlayback).toHaveBeenCalledTimes(playbackStopsAfterUnmount)
    } finally {
      vi.useRealTimers()
    }
  })

  it.each(['stale-first', 'current-first'] as const)(
    'scopes overlapping lifecycle microphone starts when they resolve %s',
    async resolveOrder => {
      const microphoneA = deferred<undefined>()
      const microphoneB = deferred<undefined>()
      const startSignals: AbortSignal[] = []
      micHandle.start
        .mockImplementationOnce(options => {
          startSignals.push(options?.signal as AbortSignal)

          return microphoneA.promise
        })
        .mockImplementationOnce(options => {
          startSignals.push(options?.signal as AbortSignal)

          return microphoneB.promise
        })

      const hook = renderHook(() =>
        useVoiceConversation({
          busy: false,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          onTranscribeAudio: vi.fn(async () => ''),
          pendingResponse: () => null
        })
      )

      let startA!: Promise<void>
      let startB!: Promise<void>
      act(() => {
        startA = hook.result.current.start()
      })
      await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(1))

      await act(async () => hook.result.current.end())
      act(() => {
        startB = hook.result.current.start()
      })
      await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(2))

      expect(startSignals[0]?.aborted).toBe(true)
      expect(startSignals[1]?.aborted).toBe(false)

      await act(async () => {
        if (resolveOrder === 'stale-first') {
          microphoneA.resolve(undefined)
          await startA
          microphoneB.resolve(undefined)
          await startB
        } else {
          microphoneB.resolve(undefined)
          await startB
          microphoneA.resolve(undefined)
          await startA
        }
      })

      expect(startSignals[1]?.aborted).toBe(false)
      expect(hook.result.current.status).toBe('listening')
    }
  )

  it('releases a microphone acquired after the conversation becomes busy', async () => {
    const microphoneStarted = deferred<undefined>()
    micHandle.start.mockImplementationOnce(() => microphoneStarted.promise)

    const hook = renderHook(
      ({ busy }: HookProps) =>
        useVoiceConversation({
          busy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          onTranscribeAudio: vi.fn(async () => ''),
          pendingResponse: () => null
        }),
      { initialProps: { busy: false } }
    )

    let starting!: Promise<void>

    act(() => {
      starting = hook.result.current.start()
    })
    await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(1))

    hook.rerender({ busy: true })
    const cancelCallsBeforeAcquisition = micHandle.cancel.mock.calls.length

    await act(async () => {
      microphoneStarted.resolve(undefined)
      await starting
    })

    expect(hook.result.current.status).toBe('idle')
    expect(micHandle.cancel).toHaveBeenCalledTimes(cancelCallsBeforeAcquisition + 1)
  })

  it('aborts a pending microphone start on unmount', async () => {
    const microphoneStarted = deferred<undefined>()
    let startSignal: AbortSignal | undefined
    micHandle.start.mockImplementationOnce(options => {
      startSignal = options?.signal

      return microphoneStarted.promise
    })

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit: vi.fn(),
        onTranscribeAudio: vi.fn(async () => ''),
        pendingResponse: () => null
      })
    )

    let startPromise!: Promise<void>
    act(() => {
      startPromise = hook.result.current.start()
    })
    await waitFor(() => expect(startSignal).toBeDefined())

    hook.unmount()
    expect(startSignal?.aborted).toBe(true)

    await act(async () => {
      microphoneStarted.resolve(undefined)
      await startPromise
    })
  })

  it('expires while microphone startup is pending without reviving listening or arming a stale timer', async () => {
    vi.useFakeTimers()
    const microphoneStarted = deferred<undefined>()
    const onIdleTimeout = vi.fn()
    micHandle.start.mockImplementationOnce(() => microphoneStarted.promise)

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 1_000,
        onIdleTimeout,
        onSubmit: vi.fn(),
        onTranscribeAudio: vi.fn(async () => ''),
        pendingResponse: () => null
      })
    )

    try {
      let startPromise!: Promise<void>
      act(() => {
        startPromise = hook.result.current.start()
      })

      await act(async () => {
        vi.advanceTimersByTime(1_000)
      })
      expect(onIdleTimeout).toHaveBeenCalledTimes(1)

      await act(async () => {
        microphoneStarted.resolve(undefined)
        await startPromise
      })

      expect(hook.result.current.status).toBe('idle')
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('ignores normal transcription that resolves after inactivity expiry', async () => {
    vi.useFakeTimers()
    const transcription = deferred<string>()
    const onIdleTimeout = vi.fn()
    const onSubmit = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 1_000,
        onIdleTimeout,
        onSubmit,
        onTranscribeAudio: vi.fn(() => transcription.promise),
        pendingResponse: () => null
      })
    )

    try {
      await act(async () => {
        await hook.result.current.start()
      })
      micHandle.stop.mockResolvedValueOnce({
        audio: new Blob(['speech'], { type: 'audio/webm' }),
        durationMs: 400,
        heardSpeech: true
      })
      act(() => hook.result.current.stopTurn())
      expect(hook.result.current.status).toBe('transcribing')

      await act(async () => {
        vi.advanceTimersByTime(1_000)
      })
      expect(onIdleTimeout).toHaveBeenCalledTimes(1)

      await act(async () => {
        transcription.resolve('too late')
        await transcription.promise
      })

      expect(onSubmit).not.toHaveBeenCalled()
      expect(hook.result.current.status).toBe('idle')
    } finally {
      vi.useRealTimers()
    }
  })

  it('ignores barge-in transcription that resolves after inactivity expiry', async () => {
    const bargeTranscription = deferred<string>()
    const onIdleTimeout = vi.fn()
    const busyChange: { current: (busy: boolean) => void } = { current: () => undefined }
    const onSubmit = vi.fn(async () => busyChange.current(true))
    let transcriptions = 0

    const hook = renderHook(
      ({ busy }: HookProps) =>
        useVoiceConversation({
          busy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          idleTimeoutMs: 200,
          onIdleTimeout,
          onSubmit,
          onTranscribeAudio: vi.fn(() =>
            transcriptions++ === 0 ? Promise.resolve('start the task') : bargeTranscription.promise
          ),
          pendingResponse: () => null
        }),
      { initialProps: { busy: false } }
    )

    busyChange.current = busy => hook.rerender({ busy })

    try {
      await act(async () => {
        await hook.result.current.start()
      })
      micHandle.stop.mockResolvedValueOnce({
        audio: new Blob(['first'], { type: 'audio/webm' }),
        durationMs: 400,
        heardSpeech: true
      })
      await act(async () => hook.result.current.stopTurn())
      await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

      const monitor = monitorCalls.at(-1)
      act(() => monitor?.onSpeech())
      hook.rerender({ busy: false })
      act(() => monitor?.onUtterance?.(new Blob(['barge'], { type: 'audio/webm' })))
      expect(hook.result.current.status).toBe('transcribing')

      await waitFor(() => expect(onIdleTimeout).toHaveBeenCalledTimes(1))

      await act(async () => {
        bargeTranscription.resolve('too late')
        await bargeTranscription.promise
      })

      expect(onSubmit).toHaveBeenCalledTimes(1)
      expect(hook.result.current.status).toBe('idle')
      expect(micHandle.start).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not arm the inactivity interval when configured with zero', async () => {
    vi.useFakeTimers()
    const onIdleTimeout = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        idleTimeoutMs: 0,
        onIdleTimeout,
        onSubmit: vi.fn(),
        onTranscribeAudio: vi.fn(async () => ''),
        pendingResponse: () => null
      })
    )

    try {
      await act(async () => {
        await hook.result.current.start()
        vi.advanceTimersByTime(60_001)
      })

      expect(onIdleTimeout).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })
})
