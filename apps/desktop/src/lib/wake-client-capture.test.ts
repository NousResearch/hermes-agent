import { afterEach, describe, expect, it, vi } from 'vitest'

import { startClientWakeCapture } from './wake-client-capture'

describe('startClientWakeCapture', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('asks Desktop for microphone access before opening the browser microphone', async () => {
    const requestMicrophoneAccess = vi.fn().mockResolvedValue(false)
    const getUserMedia = vi.fn().mockRejectedValue(new Error('getUserMedia should not run'))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess }
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })
    vi.stubGlobal('AudioContext', class {})

    await expect(startClientWakeCapture({ request: vi.fn() })).rejects.toThrow('Microphone access denied')

    expect(requestMicrophoneAccess).toHaveBeenCalledOnce()
    expect(getUserMedia).not.toHaveBeenCalled()
  })

  it('surfaces a Desktop microphone-access preflight failure', async () => {
    const requestMicrophoneAccess = vi.fn().mockRejectedValue(new Error('macOS microphone preflight failed'))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess }
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn() }
    })
    vi.stubGlobal('AudioContext', class {})

    await expect(startClientWakeCapture({ request: vi.fn() })).rejects.toThrow('macOS microphone preflight failed')

    expect(requestMicrophoneAccess).toHaveBeenCalledOnce()
  })
})
