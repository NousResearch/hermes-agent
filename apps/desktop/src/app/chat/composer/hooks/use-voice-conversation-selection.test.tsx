import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as voicePlayback from '@/lib/voice-playback'

const mocks = vi.hoisted(() => ({
  monitorStop: vi.fn(),
  monitorSpeechDuringPlayback: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  playSpeechText: vi.fn(),
  recorderCancel: vi.fn(),
  recorderStart: vi.fn(),
  recorderStop: vi.fn(),
  startSpeechStream: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'Configure speech to text',
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
vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: mocks.monitorSpeechDuringPlayback
}))
vi.mock('@/lib/voice-playback', async importOriginal => ({
  ...(await importOriginal<typeof voicePlayback>()),
  playSpeechText: mocks.playSpeechText,
  startSpeechStream: mocks.startSpeechStream
}))
vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))
vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: {
      cancel: mocks.recorderCancel,
      start: mocks.recorderStart,
      stop: mocks.recorderStop
    },
    level: 0
  })
}))

import { $voicePlayback, setVoicePlaybackState } from '@/store/voice-playback'

import { useVoiceConversation } from './use-voice-conversation'

interface PendingResponse {
  id: string
  pending: boolean
  text: string
}

interface RenderConversationOptions {
  beforeMicOpen?: () => Promise<void> | void
}

function deferred() {
  let resolve!: () => void

  const promise = new Promise<void>(resolvePromise => {
    resolve = resolvePromise
  })

  return { promise, resolve }
}

function claimPlayback(messageId: string, source: 'read-aloud' | 'voice-conversation') {
  voicePlayback.stopVoicePlayback()
  setVoicePlaybackState({
    audioElement: null,
    messageId,
    sequence: voicePlayback.getVoicePlaybackSequence(),
    source,
    status: 'preparing'
  })
}

function releasePlayback() {
  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence: voicePlayback.getVoicePlaybackSequence(),
    source: null,
    status: 'idle'
  })
}

describe('useVoiceConversation playback ownership', () => {
  let recorderOnSilence: (() => void) | null

  beforeEach(() => {
    recorderOnSilence = null
    Object.values(mocks).forEach(mock => mock.mockReset())
    mocks.monitorSpeechDuringPlayback.mockImplementation(() => mocks.monitorStop)
    mocks.playSpeechText.mockResolvedValue(true)
    mocks.recorderStart.mockImplementation(async (options: { onSilence: () => void }) => {
      recorderOnSilence = options.onSilence
    })
    mocks.recorderStop.mockResolvedValue({
      audio: new Blob(['voice']),
      heardSpeech: true
    })
    voicePlayback.stopVoicePlayback()
  })

  afterEach(() => {
    cleanup()
    voicePlayback.stopVoicePlayback()
  })

  function renderConversation(
    startSpeech: () => Promise<null | undefined>,
    options: RenderConversationOptions = {}
  ) {
    let pendingResponse: PendingResponse | null = null

    const consumePendingResponse = vi.fn(() => {
      pendingResponse = null
    })

    mocks.startSpeechStream.mockImplementation(() => {
      claimPlayback('voice-response', 'voice-conversation')

      return startSpeech()
    })

    const onSubmit = vi.fn(async () => undefined)

    const hook = renderHook(
      ({ busy }) =>
        useVoiceConversation({
          beforeMicOpen: options.beforeMicOpen,
          busy,
          consumePendingResponse,
          enabled: true,
          onSubmit,
          onTranscribeAudio: vi.fn(async () => 'spoken request'),
          pendingResponse: () => pendingResponse
        }),
      { initialProps: { busy: false } }
    )

    const beginVoiceTurn = async () => {
      await act(async () => hook.result.current.start())
      expect(hook.result.current.status).toBe('listening')
      hook.rerender({ busy: true })
      act(() => recorderOnSilence?.())
      await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('spoken request'))
      await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    }

    const submitVoiceTurn = async (response: PendingResponse) => {
      await beginVoiceTurn()

      pendingResponse = response
      hook.rerender({ busy: true })
      await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalledOnce())
      await waitFor(() => expect(hook.result.current.status).toBe('speaking'))
    }

    return {
      beginVoiceTurn,
      completeResponse: (text: string) => {
        pendingResponse = { id: 'response-1', pending: false, text }
        hook.rerender({ busy: false })
      },
      consumePendingResponse,
      hook,
      setPendingResponse: (response: PendingResponse, busy: boolean) => {
        pendingResponse = response
        hook.rerender({ busy })
      },
      submitVoiceTurn
    }
  }

  it('pauses listening while selected text owns playback, then resumes', async () => {
    const conversation = renderConversation(async () => null)

    await act(async () => conversation.hook.result.current.start())
    expect(conversation.hook.result.current.status).toBe('listening')
    expect(mocks.recorderStart).toHaveBeenCalledOnce()

    act(() => claimPlayback('selection-read-aloud', 'read-aloud'))

    await waitFor(() => expect(mocks.recorderCancel).toHaveBeenCalledOnce())
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('idle'))

    act(() => releasePlayback())

    await waitFor(() => expect(mocks.recorderStart).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('listening'))
  })

  it('revalidates external playback after beforeMicOpen and resumes exactly once', async () => {
    const firstGate = deferred()
    const secondGate = deferred()

    const beforeMicOpen = vi
      .fn()
      .mockImplementationOnce(() => firstGate.promise)
      .mockImplementationOnce(() => secondGate.promise)

    const conversation = renderConversation(async () => null, { beforeMicOpen })
    let initialStart!: Promise<void>

    await act(async () => {
      initialStart = conversation.hook.result.current.start()
      await Promise.resolve()
    })
    expect(beforeMicOpen).toHaveBeenCalledOnce()

    act(() => claimPlayback('selection-read-aloud', 'read-aloud'))
    await act(async () => {
      firstGate.resolve()
      await initialStart
    })

    expect(mocks.recorderStart).not.toHaveBeenCalled()
    expect(conversation.hook.result.current.status).toBe('idle')

    act(() => releasePlayback())
    await waitFor(() => expect(beforeMicOpen).toHaveBeenCalledTimes(2))

    await act(async () => {
      conversation.hook.rerender({ busy: false })
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(beforeMicOpen).toHaveBeenCalledTimes(2)
    expect(mocks.recorderStart).not.toHaveBeenCalled()

    await act(async () => {
      secondGate.resolve()
      await secondGate.promise
    })

    await waitFor(() => expect(mocks.recorderStart).toHaveBeenCalledOnce())
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('listening'))

    await act(async () => {
      conversation.hook.rerender({ busy: false })
      await Promise.resolve()
    })
    expect(beforeMicOpen).toHaveBeenCalledTimes(2)
    expect(mocks.recorderStart).toHaveBeenCalledOnce()
  })

  it('does not start reply speech over selected playback that began while thinking', async () => {
    const conversation = renderConversation(async () => null)

    await conversation.beginVoiceTurn()
    act(() => claimPlayback('selection-read-aloud', 'read-aloud'))

    await waitFor(() => expect(conversation.hook.result.current.status).toBe('idle'))
    act(() => conversation.completeResponse('Completed voice reply.'))

    expect(mocks.startSpeechStream).not.toHaveBeenCalled()
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })

    act(() => releasePlayback())

    await waitFor(() => expect(mocks.recorderStart).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('listening'))
  })

  it('discards a late interrupted response without discarding the next voice turn', async () => {
    const conversation = renderConversation(async () => null)

    await conversation.beginVoiceTurn()
    conversation.consumePendingResponse.mockClear()
    act(() => claimPlayback('selection-read-aloud', 'read-aloud'))

    await waitFor(() => expect(conversation.hook.result.current.status).toBe('idle'))
    expect(conversation.consumePendingResponse).not.toHaveBeenCalled()

    act(() => conversation.completeResponse('Late interrupted reply.'))
    await waitFor(() => expect(conversation.consumePendingResponse).toHaveBeenCalledOnce())
    expect(mocks.startSpeechStream).not.toHaveBeenCalled()

    act(() => releasePlayback())
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('listening'))

    conversation.hook.rerender({ busy: true })
    act(() => recorderOnSilence?.())
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('thinking'))

    act(() =>
      conversation.setPendingResponse(
        { id: 'response-2', pending: false, text: 'Fresh response for the next turn.' },
        true
      )
    )

    await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalledOnce())
  })

  it('does not let stale fallback polling replace newer selected-text playback', async () => {
    // Keep a live speech session open. Returning null makes current main treat the
    // startSpeechStream sequence bump as "stopped during start" and never enter
    // speaking long enough for this ownership race to be exercised.
    const liveSession = {
      append: vi.fn(),
      done: new Promise<'done'>(() => undefined),
      fallbackMaxChars: () => null,
      finish: vi.fn()
    }
    const conversation = renderConversation(async () => liveSession)

    await conversation.submitVoiceTurn({ id: 'response-1', pending: true, text: 'Partial' })
    act(() => {
      claimPlayback('selection-read-aloud', 'read-aloud')
      conversation.completeResponse('Completed voice reply.')
    })

    await act(async () => {
      await new Promise(resolve => window.setTimeout(resolve, 300))
    })

    expect(mocks.playSpeechText).not.toHaveBeenCalled()
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })
  })

  it('cleans up local speaking state when stream startup is superseded', async () => {
    let resolveStream!: (value: undefined) => void

    const conversation = renderConversation(() => new Promise<undefined>(resolve => (resolveStream = resolve)))

    await conversation.submitVoiceTurn({ id: 'response-1', pending: true, text: 'Partial' })
    act(() => claimPlayback('selection-read-aloud', 'read-aloud'))
    await act(async () => resolveStream(undefined))

    expect(conversation.hook.result.current.status).toBe('idle')
    expect(mocks.monitorStop).toHaveBeenCalled()
    expect($voicePlayback.get()).toMatchObject({
      messageId: 'selection-read-aloud',
      status: 'preparing'
    })
  })

  it('resumes listening after selected playback releases ownership', async () => {
    let resolveStream!: (value: undefined) => void

    const conversation = renderConversation(() => new Promise<undefined>(resolve => (resolveStream = resolve)))

    await conversation.submitVoiceTurn({ id: 'response-1', pending: true, text: 'Partial' })
    expect(mocks.recorderStart).toHaveBeenCalledOnce()

    act(() => {
      claimPlayback('selection-read-aloud', 'read-aloud')
      conversation.completeResponse('Completed voice reply.')
    })
    await act(async () => resolveStream(undefined))

    expect(conversation.hook.result.current.status).toBe('idle')

    act(() => releasePlayback())

    await waitFor(() => expect(mocks.recorderStart).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(conversation.hook.result.current.status).toBe('listening'))
  })
})
