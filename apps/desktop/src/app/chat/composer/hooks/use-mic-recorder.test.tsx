import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MicRecorderErrorCopy, useMicRecorder } from './use-mic-recorder'

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'access denied',
  microphoneConstraintsUnsupported: 'constraints unsupported',
  microphoneInUse: 'in use',
  microphonePermissionDenied: 'permission denied',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'no microphone'
}

function fakeTrack() {
  const track = {
    readyState: 'live' as MediaStreamTrack['readyState'],
    stop: vi.fn(() => {
      track.readyState = 'ended'
    })
  }

  return track
}

// One entry per getUserMedia call, each with its own track, so a test can end
// the first stream's track without touching the replacement.
let streamTracks: ReturnType<typeof fakeTrack>[] = []
let getUserMedia: ReturnType<typeof vi.fn>
let recorderInstances: FakeRecorder[] = []

// Minimal MediaRecorder stand-in: jsdom has none. Only the surface the hook
// touches (state, start/stop, ondataavailable/onstop/onerror) is modelled.
class FakeRecorder {
  state: 'inactive' | 'recording' = 'inactive'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((event: unknown) => void) | null = null
  start = vi.fn(() => {
    this.state = 'recording'
  })

  stop = vi.fn(() => {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['audio'], { type: 'audio/webm' }) })
    this.onstop?.()
  })

  constructor() {
    recorderInstances.push(this)
  }

  static isTypeSupported() {
    return true
  }
}

beforeEach(() => {
  streamTracks = []
  recorderInstances = []
  getUserMedia = vi.fn(async () => {
    const track = fakeTrack()
    streamTracks.push(track)

    return { getTracks: () => [track] } as unknown as MediaStream
  })

  vi.stubGlobal('MediaRecorder', FakeRecorder)
  vi.stubGlobal('navigator', {
    ...navigator,
    mediaDevices: { getUserMedia }
  })
  vi.stubGlobal('AudioContext', undefined)
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
    requestMicrophoneAccess: vi.fn(async () => true)
  }
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const started = () => waitFor(() => expect(recorderInstances.length).toBeGreaterThan(0))

describe('useMicRecorder stream lifecycle', () => {
  // The mic re-arm fix rests on this: a normal stop between turns keeps the
  // MediaStream open, so the next turn skips the getUserMedia round-trip that
  // was costing seconds after the assistant finished speaking.
  it('keeps the stream open across a normal stop and reuses it on the next start', async () => {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    await started()

    await act(async () => {
      await result.current.handle.stop()
    })

    expect(streamTracks[0].stop).not.toHaveBeenCalled()
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    await act(async () => {
      await result.current.handle.start()
    })

    // Second turn re-arms off the retained stream.
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(streamTracks[0].stop).not.toHaveBeenCalled()
  })

  it('releases the stream on cancel', async () => {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    await started()

    act(() => result.current.handle.cancel())

    expect(streamTracks[0].stop).toHaveBeenCalled()
  })

  it('releases the stream on unmount', async () => {
    const { result, unmount } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    await started()

    unmount()

    expect(streamTracks[0].stop).toHaveBeenCalled()
  })

  it('releases the stream when the recorder errors', async () => {
    const onError = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onError })
    })
    await started()

    act(() => recorderInstances[0].onerror?.(new Event('error')))

    expect(streamTracks[0].stop).toHaveBeenCalled()
    expect(onError).toHaveBeenCalled()
  })

  // A stream whose track died while idle (device unplugged, OS revoked it) must
  // not be reused: the hook re-requests rather than arming a dead recorder.
  it('re-requests the stream when the retained tracks are no longer live', async () => {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    await started()
    await act(async () => {
      await result.current.handle.stop()
    })

    streamTracks[0].readyState = 'ended'

    await act(async () => {
      await result.current.handle.start()
    })

    expect(getUserMedia).toHaveBeenCalledTimes(2)
  })
})
