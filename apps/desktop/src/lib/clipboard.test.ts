import { afterEach, describe, expect, it, vi } from 'vitest'

import { createBrowserClipboardBridge } from './browser-clipboard'
import { installClipboardShim, writeClipboardText } from './clipboard'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText }
  })
}

afterEach(() => {
  Reflect.deleteProperty(desktopWindow, 'hermesDesktop')
  Reflect.deleteProperty(navigator, 'clipboard')
  vi.restoreAllMocks()
  delete document.documentElement.dataset.hermesDesktopHost
})

describe('installClipboardShim', () => {
  it('keeps a successful native write on the trusted path', async () => {
    const nativeWrite = vi.fn().mockResolvedValue(undefined)
    const ipcWrite = vi.fn().mockResolvedValue(true)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: ipcWrite } as unknown as Window['hermesDesktop']

    installClipboardShim()
    await navigator.clipboard.writeText('payload')

    expect(nativeWrite).toHaveBeenCalledWith('payload')
    expect(ipcWrite).not.toHaveBeenCalled()
  })

  it('uses Electron IPC only after the native write fails', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('lost focus'))
    const ipcWrite = vi.fn().mockResolvedValue(true)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: ipcWrite } as unknown as Window['hermesDesktop']

    installClipboardShim()
    await navigator.clipboard.writeText('payload')

    expect(nativeWrite).toHaveBeenCalledWith('payload')
    expect(ipcWrite).toHaveBeenCalledWith('payload')
  })
})

describe('writeClipboardText', () => {
  it('attempts native and IPC once when an installed Electron shim fails', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('lost focus'))
    const bridgeWrite = vi.fn().mockResolvedValue(false)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: bridgeWrite } as unknown as Window['hermesDesktop']
    installClipboardShim()
    await expect(writeClipboardText('payload')).rejects.toThrow()
    expect(nativeWrite).toHaveBeenCalledTimes(1)
    expect(bridgeWrite).toHaveBeenCalledTimes(1)
  })

  it('does not retry the same native API through the installed browser bridge', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('permission denied'))
    installClipboard(nativeWrite)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    desktopWindow.hermesDesktop = createBrowserClipboardBridge({
      saveBuffer: async () => ''
    }) as Window['hermesDesktop']
    installClipboardShim()
    await expect(writeClipboardText('payload')).rejects.toThrow('permission denied')
    expect(nativeWrite).toHaveBeenCalledTimes(1)
  })

  it('reports an unavailable bridge as a failed copy', async () => {
    const bridgeWrite = vi.fn().mockResolvedValue(false)
    desktopWindow.hermesDesktop = { writeClipboard: bridgeWrite } as unknown as Window['hermesDesktop']

    await expect(writeClipboardText('payload')).rejects.toThrow('Clipboard write is unavailable')
  })
})
