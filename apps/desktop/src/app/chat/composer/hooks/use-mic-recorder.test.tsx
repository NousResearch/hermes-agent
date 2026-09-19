import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MicRecorderErrorCopy, useMicRecorder } from './use-mic-recorder'

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: '',
  microphoneConstraintsUnsupported: '',
  microphoneInUse: '',
  microphonePermissionDenied: '',
  microphoneStartFailed: '',
  microphoneUnsupported: '',
  noMicrophone: ''
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('useMicRecorder acquisition ownership', () => {
  const originalMediaRecorder = globalThis.MediaRecorder

  beforeEach(() => {
    Object.defineProperty(globalThis, 'MediaRecorder', { configurable: true, value: vi.fn() })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess: vi.fn(async () => true) }
    })
  })

  afterEach(() => {
    Object.defineProperty(globalThis, 'MediaRecorder', { configurable: true, value: originalMediaRecorder })
    vi.restoreAllMocks()
  })

  it('acquires and starts recording after StrictMode effect replay', async () => {
    const starts = vi.fn()
    const constructors = vi.fn()

    class MediaRecorderMock {
      static isTypeSupported() {
        return false
      }

      mimeType = ''
      ondataavailable = null
      onerror = null
      onstop = null
      start = starts

      constructor() {
        constructors()
      }
    }

    Object.defineProperty(globalThis, 'MediaRecorder', { configurable: true, value: MediaRecorderMock })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [] })) }
    })
    const hook = renderHook(() => useMicRecorder(copy), { reactStrictMode: true })

    await act(async () => hook.result.current.handle.start())

    expect(constructors).toHaveBeenCalledTimes(1)
    expect(starts).toHaveBeenCalledTimes(1)
    expect(hook.result.current.recording).toBe(true)
  })

  it('stops every late getUserMedia track after cancel', async () => {
    const acquisition = deferred<MediaStream>()
    const stop = vi.fn()
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(() => acquisition.promise) }
    })
    const hook = renderHook(() => useMicRecorder(copy))

    let started!: Promise<void>
    act(() => {
      started = hook.result.current.handle.start()
    })
    await act(async () => Promise.resolve())
    act(() => hook.result.current.handle.cancel())
    acquisition.resolve({ getTracks: () => [{ stop }] } as unknown as MediaStream)
    await act(async () => started)

    expect(stop).toHaveBeenCalledTimes(1)
    expect(globalThis.MediaRecorder).not.toHaveBeenCalled()
  })

  it('stops every late getUserMedia track after unmount', async () => {
    const acquisition = deferred<MediaStream>()
    const stop = vi.fn()
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(() => acquisition.promise) }
    })
    const hook = renderHook(() => useMicRecorder(copy))

    let started!: Promise<void>
    act(() => {
      started = hook.result.current.handle.start()
    })
    await act(async () => Promise.resolve())
    hook.unmount()
    acquisition.resolve({ getTracks: () => [{ stop }] } as unknown as MediaStream)
    await act(async () => started)

    expect(stop).toHaveBeenCalledTimes(1)
    expect(globalThis.MediaRecorder).not.toHaveBeenCalled()
  })
})
