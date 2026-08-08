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

// ---------------------------------------------------------------------------
// Mute semantics — #74337 class: muting mid-conversation must be respected
// (no re-arm while muted) AND reversible (unmute returns the loop to
// listening), with the mic never wedging into a stuck state after playback.
// ---------------------------------------------------------------------------

function renderWithReply() {
  let reply: { id: string; pending: boolean; text: string } | null = null
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const hook = renderHook(
    ({ busy }: HookProps) =>
      useVoiceConversation({
        busy,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit: async () => {
          // Mirrors the real app: submitting a turn makes the agent busy, and
          // the reply only appears once the turn ends (busy flips false).
          reply = { id: 'reply-1', pending: true, text: 'Hello back' }
          onBusyChange.current(true)
        },
        onTranscribeAudio: async () => 'Hello',
        pendingResponse: () => (busy ? null : reply)
      }),
    { initialProps: { busy: false } }
  )

  onBusyChange.current = busy => hook.rerender({ busy })

  return {
    completeReply() {
      if (reply) {
        reply = { ...reply, pending: false }
      }
    },
    finishTurn() {
      onBusyChange.current(false)
    },
    hook
  }
}

async function beginTurn(hook: ReturnType<typeof renderWithReply>['hook']) {
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

describe('useVoiceConversation mute semantics', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('muting while listening cancels the recorder and keeps the mic off until unmute', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    act(() => {
      hook.result.current.toggleMute()
    })

    expect(hook.result.current.muted).toBe(true)
    expect(hook.result.current.status).toBe('idle')
    expect(micHandle.cancel).toHaveBeenCalled()

    // The drive loop must not re-open the mic while muted.
    await act(async () => {
      await new Promise(resolve => window.setTimeout(resolve, 50))
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    // Unmuting re-arms the loop without a manual stop/start.
    act(() => {
      hook.result.current.toggleMute()
    })

    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    expect(micHandle.start).toHaveBeenCalledTimes(2)
  })

  it('a reply completing while muted is never spoken and unmute re-arms into the next turn', async () => {
    const { completeReply, finishTurn, hook } = renderWithReply()

    await beginTurn(hook)

    // Mute during generation; the turn's reply is held, not spoken.
    act(() => {
      hook.result.current.toggleMute()
    })
    expect(hook.result.current.muted).toBe(true)
    expect(hook.result.current.status).toBe('idle')

    // The turn ends and the reply becomes available — but muted gates speech.
    await act(async () => {
      finishTurn()
    })
    expect(hook.result.current.status).toBe('idle')
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    // Unmute: the held reply plays (fallback path), then the mic re-arms.
    act(() => {
      hook.result.current.toggleMute()
    })

    await waitFor(() => expect(hook.result.current.status).toBe('speaking'))
    completeReply()
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    expect(micHandle.start).toHaveBeenCalledTimes(2)
  })

  it('muting during reply playback does not wedge the loop; unmute returns it to listening', async () => {
    const { completeReply, finishTurn, hook } = renderWithReply()

    await beginTurn(hook)
    // Turn ends → the (pending) reply lands → live speech opens and the
    // fallback poll holds 'speaking' until the reply completes.
    await act(async () => {
      finishTurn()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('speaking'))

    // User mutes mid-playback to silence the rest of the reply.
    act(() => {
      hook.result.current.toggleMute()
    })
    expect(hook.result.current.muted).toBe(true)

    // Reply completes while muted → settle runs, but no re-arm while muted.
    completeReply()
    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    await act(async () => {
      await new Promise(resolve => window.setTimeout(resolve, 50))
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    // Unmute returns the loop to listening — no manual stop/start needed.
    act(() => {
      hook.result.current.toggleMute()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))
    expect(micHandle.start).toHaveBeenCalledTimes(2)
  })
})
