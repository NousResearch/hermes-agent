import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useMicRecorder } from './use-mic-recorder'

const copy = {
  microphoneAccessDenied: 'denied',
  microphoneInUse: 'in use',
  microphoneUnsupported: 'unsupported',
  permissionDenied: 'permission',
  recordingFailed: 'failed'
} as never

describe('useMicRecorder async cancellation', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('closes a stream that arrives after cancel and never creates a recorder', async () => {
    let resolveStream: (stream: MediaStream) => void = () => undefined

    const getUserMedia = vi.fn(
      () =>
        new Promise<MediaStream>(resolve => {
          resolveStream = resolve
        })
    )

    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream
    const Recorder = vi.fn()
    Object.assign(Recorder, { isTypeSupported: vi.fn(() => true) })

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess: vi.fn().mockResolvedValue(true) }
    })
    vi.stubGlobal('MediaRecorder', Recorder)

    const { result } = renderHook(() => useMicRecorder(copy))
    let starting: Promise<boolean> = Promise.resolve(false)

    await act(async () => {
      starting = result.current.handle.start()
      await Promise.resolve()
      result.current.handle.cancel()
      resolveStream(stream)
      await starting
    })

    expect(stopTrack).toHaveBeenCalledTimes(1)
    expect(Recorder).not.toHaveBeenCalled()
    expect(result.current.recording).toBe(false)
  })

  it('ignores a getUserMedia rejection that arrives after cancel', async () => {
    let rejectStream: (error: Error) => void = () => undefined

    const getUserMedia = vi.fn(
      () =>
        new Promise<MediaStream>((_resolve, reject) => {
          rejectStream = reject
        })
    )

    const Recorder = vi.fn()
    Object.assign(Recorder, { isTypeSupported: vi.fn(() => true) })

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess: vi.fn().mockResolvedValue(true) }
    })
    vi.stubGlobal('MediaRecorder', Recorder)

    const { result } = renderHook(() => useMicRecorder(copy))
    let starting: Promise<boolean> = Promise.resolve(false)

    await act(async () => {
      starting = result.current.handle.start()
      await Promise.resolve()
      result.current.handle.cancel()
      rejectStream(new Error('stale failure'))
      await expect(starting).resolves.toBe(false)
    })

    expect(Recorder).not.toHaveBeenCalled()
    expect(result.current.recording).toBe(false)
  })

  it.each([
    ['a denied permission result', false],
    ['a rejected permission request', new Error('stale permission failure')]
  ])('ignores %s that arrives after cancel', async (_label, outcome) => {
    let settlePermission: () => void = () => undefined

    const requestMicrophoneAccess = vi.fn(
      () =>
        new Promise<boolean>((resolve, reject) => {
          settlePermission = () => (outcome instanceof Error ? reject(outcome) : resolve(outcome))
        })
    )

    const getUserMedia = vi.fn()
    const Recorder = vi.fn()
    Object.assign(Recorder, { isTypeSupported: vi.fn(() => true) })

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess }
    })
    vi.stubGlobal('MediaRecorder', Recorder)

    const { result } = renderHook(() => useMicRecorder(copy))
    let starting: Promise<boolean> = Promise.resolve(false)

    await act(async () => {
      starting = result.current.handle.start()
      result.current.handle.cancel()
      settlePermission()
      await expect(starting).resolves.toBe(false)
    })

    expect(getUserMedia).not.toHaveBeenCalled()
    expect(Recorder).not.toHaveBeenCalled()
    expect(result.current.recording).toBe(false)
  })
})
