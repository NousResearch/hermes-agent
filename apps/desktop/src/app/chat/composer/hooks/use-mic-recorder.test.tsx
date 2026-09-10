import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MicRecorderErrorCopy, useMicRecorder } from './use-mic-recorder'

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'access denied',
  microphoneConstraintsUnsupported: 'unsupported constraints',
  microphoneInUse: 'microphone in use',
  microphonePermissionDenied: 'permission denied',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'no microphone'
}

class TestRecorder {
  static instances: TestRecorder[] = []
  static isTypeSupported = () => true
  state = 'inactive'
  onstop: (() => void) | null = null
  ondataavailable: unknown = null
  onerror: unknown = null
  start = vi.fn(() => {
    this.state = 'recording'
  })
  stop = vi.fn(() => {
    this.state = 'inactive'
    this.onstop?.()
  })

  constructor(readonly stream: MediaStream) {
    TestRecorder.instances.push(this)
  }
}

function audioStream() {
  const stop = vi.fn()

  return { stop, stream: { getTracks: () => [{ stop }] } as unknown as MediaStream }
}

describe('useMicRecorder acquisition ownership', () => {
  beforeEach(() => {
    TestRecorder.instances.length = 0
    vi.stubGlobal('MediaRecorder', TestRecorder)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('releases an obsolete microphone grant without replacing or stopping its successor', async () => {
    const obsolete = audioStream()
    const current = audioStream()
    let grantObsolete!: (stream: MediaStream) => void

    const getUserMedia = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            grantObsolete = resolve
          })
      )
      .mockResolvedValueOnce(current.stream)

    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } })
    const hook = renderHook(() => useMicRecorder(copy))
    let first!: Promise<void>
    await act(async () => {
      first = hook.result.current.handle.start()
    })
    act(() => {
      hook.result.current.handle.cancel()
    })
    await act(async () => {
      await hook.result.current.handle.start()
    })
    expect(hook.result.current.recording).toBe(true)

    await act(async () => {
      grantObsolete(obsolete.stream)
      await first
    })

    expect(obsolete.stop).toHaveBeenCalledOnce()
    expect(current.stop).not.toHaveBeenCalled()
    expect(TestRecorder.instances).toHaveLength(1)
    expect(TestRecorder.instances[0].stream).toBe(current.stream)
    expect(hook.result.current.recording).toBe(true)
    act(() => {
      hook.result.current.handle.cancel()
    })
    expect(current.stop).toHaveBeenCalledOnce()
  })

  it('does not open capture after an unmounted native permission request resolves', async () => {
    let grantPermission!: (allowed: boolean) => void

    const requestMicrophoneAccess = vi.fn(
      () =>
        new Promise<boolean>(resolve => {
          grantPermission = resolve
        })
    )

    const getUserMedia = vi.fn().mockResolvedValue(audioStream().stream)
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess }
    })
    const hook = renderHook(() => useMicRecorder(copy))
    let starting!: Promise<void>
    act(() => {
      starting = hook.result.current.handle.start()
    })
    hook.unmount()
    await act(async () => {
      grantPermission(true)
      await starting
    })
    expect(getUserMedia).not.toHaveBeenCalled()
  })
})
