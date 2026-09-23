import { describe, expect, it, vi } from 'vitest'

import { createDesktopWindowEventsRuntime } from './desktop-window-events-runtime'

function createRuntime() {
  const send = vi.fn()

  const mainWindow = {
    isDestroyed: () => false,
    isFullScreen: () => false,
    isMinimized: () => false,
    isVisible: () => true,
    webContents: { isDestroyed: () => false, send }
  }

  const electronWebContents = { getFocusedWebContents: vi.fn(() => null as any) }

  const runtime = createDesktopWindowEventsRuntime({
    DARWIN_MAJOR: 0,
    IS_MAC: false,
    IS_WINDOWS: true,
    IS_WSL: false,
    WINDOW_BUTTON_POSITION: { x: 0, y: 0 },
    computeNativeOverlayWidth: () => 0,
    electronWebContents,
    getMainWindow: () => mainWindow as any,
    windowControlState: () => ({})
  })

  return { electronWebContents, runtime, send }
}

describe('desktop window events', () => {
  it('routes navigation to a focused webview without navigating the host', () => {
    const { electronWebContents, runtime, send } = createRuntime()
    const goBack = vi.fn()

    electronWebContents.getFocusedWebContents.mockReturnValue({
      getType: () => 'webview',
      isDestroyed: () => false,
      navigationHistory: { canGoBack: () => true, goBack }
    })
    runtime.sendPreviewNavCommand('back')

    expect(goBack).toHaveBeenCalledOnce()
    expect(send).not.toHaveBeenCalled()
  })

  it('forwards navigation to the renderer when focus is in app chrome', () => {
    const { runtime, send } = createRuntime()

    runtime.sendPreviewNavCommand('forward')

    expect(send).toHaveBeenCalledWith('hermes:preview-nav', 'forward')
  })
})
