import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useMicRecorder } from './use-mic-recorder'
import type { MicRecorderErrorCopy, MicRecording } from './use-mic-recorder'

// Streaming STT plumbing: when the gateway supports it the recorder opens a
// stream, surfaces live partial text, and hands the caller a `transcript`
// promise; when it does not, the recorder is byte-for-byte the old one-shot
// file recorder (no `transcript`).

const mocks = vi.hoisted(() => ({ open: vi.fn() }))

vi.mock('@/lib/transcription-stream', () => ({ openTranscriptionStream: mocks.open }))

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'denied',
  microphoneConstraintsUnsupported: 'constraints',
  microphoneInUse: 'in use',
  microphonePermissionDenied: 'permission',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'none'
}

function fakeStream(): MediaStream {
  const track = { stop: vi.fn() }

  return { getAudioTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream
}

class FakeMediaRecorder {
  static isTypeSupported = vi.fn(() => true)
  mimeType = 'audio/webm'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onerror: ((event: unknown) => void) | null = null
  onstop: (() => void) | null = null
  state = 'inactive'
  start() {
    this.state = 'recording'
  }
  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['chunk'], { type: 'audio/webm' }) })
    this.onstop?.()
  }
}

beforeEach(() => {
  mocks.open.mockReset()
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => fakeStream()) }
  })
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

describe('useMicRecorder streaming STT', () => {
  it('surfaces partial transcripts and returns the streaming transcript on stop', async () => {
    let onPartial: ((text: string) => void) | null = null
    const attach = vi.fn(async () => undefined)
    const cancel = vi.fn()
    const finish = vi.fn(async () => 'streamed words')

    mocks.open.mockImplementation(async (callback: (text: string) => void) => {
      onPartial = callback

      return { attach, cancel, finish }
    })

    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })

    expect(result.current.recording).toBe(true)
    expect(attach).toHaveBeenCalledTimes(1)

    act(() => onPartial?.('streamed'))
    expect(result.current.partialTranscript).toBe('streamed')

    let recording: MicRecording | null = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).not.toBeNull()
    expect(recording!.transcript).toBeDefined()
    await expect(recording!.transcript).resolves.toBe('streamed words')
    expect(finish).toHaveBeenCalledTimes(1)
  })

  it('keeps the file-based recorder (no transcript) when streaming is unavailable', async () => {
    mocks.open.mockResolvedValue(null)

    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })

    let recording: MicRecording | null = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).not.toBeNull()
    expect(recording!.transcript).toBeUndefined()
    expect(result.current.partialTranscript).toBe('')
  })
})
