import { afterEach, describe, expect, it, vi } from 'vitest'

import { installClipboardShim } from './clipboard'

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
