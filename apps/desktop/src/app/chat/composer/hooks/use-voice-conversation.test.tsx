import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as VoicePlayback from '@/lib/voice-playback'
import { setVoicePlaybackState } from '@/store/voice-playback'

import { type ConversationStatus, useVoiceConversation } from './use-voice-conversation'

const recorder = vi.hoisted(() => ({
  cancel: vi.fn(),
  start: vi.fn(),
  stop: vi.fn()
}))

const bargeMonitor = vi.hoisted(() => ({
  callbacks: null as null | { onSpeech: () => void; onUtterance?: (audio: Blob | null) => void },
  cleanup: vi.fn()
}))

const speechStream = vi.hoisted(() => ({
  resolveDone: (_outcome: 'done' | 'fallback'): void => undefined,
  start: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'Configure speech-to-text',
          couldNotStartSession: 'Could not start voice session',
          microphoneFailed: 'Microphone failed',
          playbackFailed: 'Playback failed',
          transcriptionFailed: 'Transcription failed',
          unavailable: 'Voice unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: vi.fn(
    (callbacks: { onSpeech: () => void; onUtterance?: (audio: Blob | null) => void }) => {
      bargeMonitor.callbacks = callbacks

      return bargeMonitor.cleanup
    }
  )
}))

vi.mock('@/lib/voice-playback', async importOriginal => {
  const actual = await importOriginal<typeof VoicePlayback>()

  return {
    ...actual,
    startSpeechStream: speechStream.start
  }
})

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: recorder,
    level: 0
  })
}))

function playback(status: 'idle' | 'preparing' | 'speaking') {
  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence: 0,
    source: status === 'idle' ? null : 'voice-conversation',
    status
  })
}

function Harness({
  busy = false,
  enabled,
  onStatus,
  onSubmit = () => undefined,
  onTranscribeAudio = async () => 'hello',
  pendingResponse = () => null
}: {
  busy?: boolean
  enabled: boolean
  onStatus?: (status: ConversationStatus) => void
  onSubmit?: (text: string) => boolean | Promise<boolean | void> | void
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  pendingResponse?: () => { id: string; pending: boolean; text: string } | null
}) {
  const conversation = useVoiceConversation({
    busy,
    consumePendingResponse: vi.fn(),
    enabled,
    onSubmit,
    onTranscribeAudio,
    pendingResponse
  })

  onStatus?.(conversation.status)

  return <div data-status={conversation.status} data-testid="voice-status" />
}

describe('useVoiceConversation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    playback('idle')
    recorder.start.mockResolvedValue(undefined)
    recorder.stop.mockResolvedValue({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      durationMs: 1200,
      heardSpeech: true
    })
    speechStream.start.mockImplementation(async () => ({
      append: vi.fn(),
      finish: vi.fn(),
      done: new Promise<'done' | 'fallback'>(resolve => {
        speechStream.resolveDone = resolve
      })
    }))
  })

  afterEach(() => {
    cleanup()
    playback('idle')
  })

  it('waits for active playback before starting the microphone', async () => {
    playback('speaking')

    const { rerender } = render(<Harness enabled={false} />)

    rerender(<Harness enabled />)

    await new Promise(resolve => window.setTimeout(resolve, 0))
    expect(recorder.start).not.toHaveBeenCalled()

    act(() => playback('idle'))

    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))
  })

  it('does not enter the thinking loop when submit rejects the transcript', async () => {
    let onSilence: (() => void) | undefined
    recorder.start.mockImplementation(async options => {
      onSilence = options?.onSilence
    })
    const onSubmit = vi.fn(() => false)
    const statuses: ConversationStatus[] = []

    const { rerender } = render(<Harness enabled={false} onStatus={status => statuses.push(status)} />)

    rerender(<Harness enabled onStatus={status => statuses.push(status)} onSubmit={onSubmit} />)

    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))

    await act(async () => {
      onSilence?.()
      await Promise.resolve()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('hello'))
    expect(statuses).not.toContain('thinking')
  })

  it('leaves the barge-in loop idle when the captured utterance submit is rejected', async () => {
    let onSilence: (() => void) | undefined
    recorder.start.mockImplementation(async options => {
      onSilence = options?.onSilence
    })

    const onSubmit = vi.fn().mockResolvedValueOnce(true).mockResolvedValueOnce(false)
    const pendingResponse = vi.fn(() => null as { id: string; pending: boolean; text: string } | null)

    const { getByTestId, rerender } = render(
      <Harness enabled={false} onSubmit={onSubmit} pendingResponse={pendingResponse} />
    )

    rerender(<Harness enabled onSubmit={onSubmit} pendingResponse={pendingResponse} />)
    await waitFor(() => expect(recorder.start).toHaveBeenCalledTimes(1))

    rerender(<Harness busy enabled onSubmit={onSubmit} pendingResponse={pendingResponse} />)

    await act(async () => {
      onSilence?.()
      await Promise.resolve()
    })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('hello'))

    pendingResponse.mockReturnValue({ id: 'assistant-1', pending: true, text: 'Streaming reply' })
    rerender(<Harness busy enabled onSubmit={onSubmit} pendingResponse={pendingResponse} />)
    await waitFor(() => expect(bargeMonitor.callbacks).not.toBeNull())

    await act(async () => {
      bargeMonitor.callbacks?.onSpeech()
      bargeMonitor.callbacks?.onUtterance?.(new Blob(['barge'], { type: 'audio/webm' }))
      await Promise.resolve()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(2))
    expect(getByTestId('voice-status').getAttribute('data-status')).toBe('idle')

    speechStream.resolveDone('done')
  })
})
