import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useMicRecorder } from './use-mic-recorder'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const initialMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

function installDesktopBridge(result: Awaited<ReturnType<Window['hermesDesktop']['requestMicrophoneAccess']>> | boolean) {
  desktopWindow.hermesDesktop = {
    requestMicrophoneAccess: vi.fn().mockResolvedValue(result)
  } as unknown as Window['hermesDesktop']
}

function installMediaDevices(getUserMedia = vi.fn()) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia }
  })

  return getUserMedia
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }

  if (initialMediaDevices) {
    Object.defineProperty(navigator, 'mediaDevices', initialMediaDevices)
  } else {
    Reflect.deleteProperty(navigator, 'mediaDevices')
  }
})

describe('useMicRecorder', () => {
  it('shows macOS settings guidance when microphone permission is denied', async () => {
    const getUserMedia = installMediaDevices()
    installDesktopBridge({ granted: false, settingsOpened: true, status: 'denied' })
    vi.stubGlobal('MediaRecorder', class TestMediaRecorder {})

    const { result } = renderHook(() => useMicRecorder())

    await expect(result.current.handle.start()).rejects.toThrow(
      'System Settings was opened at Privacy & Security > Microphone'
    )
    expect(getUserMedia).not.toHaveBeenCalled()
  })

  it('keeps the legacy denial message for boolean bridge results', async () => {
    const getUserMedia = installMediaDevices()
    installDesktopBridge(false)
    vi.stubGlobal('MediaRecorder', class TestMediaRecorder {})

    const { result } = renderHook(() => useMicRecorder())

    await expect(result.current.handle.start()).rejects.toThrow('Microphone permission was denied.')
    expect(getUserMedia).not.toHaveBeenCalled()
  })
})
