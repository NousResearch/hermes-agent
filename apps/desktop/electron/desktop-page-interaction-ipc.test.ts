import { describe, expect, it, vi } from 'vitest'

import { registerDesktopPageInteractionIpc } from './desktop-page-interaction-ipc'

describe('desktop page interaction IPC', () => {
  it('keeps find requests scoped to the sender and removes its forwarder on destruction', async () => {
    const handles = new Map<string, (...args: any[]) => any>()
    const uninstall = vi.fn()
    const installFoundInPageForwarder = vi.fn(() => uninstall)
    const performFindAfterIndexingStarted = vi.fn(async () => {})
    const sender = { id: 42, once: vi.fn(), isDestroyed: () => false }
    const window = { isDestroyed: () => false, webContents: sender }

    registerDesktopPageInteractionIpc({
      BrowserWindow: { fromWebContents: () => window },
      installFoundInPageForwarder,
      ipcMain: { handle: (name: string, handler: (...args: any[]) => any) => handles.set(name, handler) },
      openExternalUrl: () => true,
      openPreviewInBrowser: async () => true,
      performFindAfterIndexingStarted,
      reachablePreviewUrl: async () => true,
      stopFind: vi.fn()
    })

    const find = handles.get('hermes:find-in-page')!

    expect(await find({ sender }, 'first', {})).toEqual({ count: 0 })
    expect(await find({ sender }, 'second', {})).toEqual({ count: 0 })
    expect(installFoundInPageForwarder).toHaveBeenCalledOnce()
    expect(performFindAfterIndexingStarted).toHaveBeenCalledTimes(2)

    const destroy = sender.once.mock.calls.find(([name]) => name === 'destroyed')![1]

    destroy()

    expect(uninstall).toHaveBeenCalledOnce()
  })

  it('rejects invalid external URLs through the native open channel', () => {
    const handles = new Map<string, (...args: any[]) => any>()

    registerDesktopPageInteractionIpc({
      BrowserWindow: { fromWebContents: () => null },
      installFoundInPageForwarder: vi.fn(),
      ipcMain: { handle: (name: string, handler: (...args: any[]) => any) => handles.set(name, handler) },
      openExternalUrl: () => false,
      openPreviewInBrowser: async () => false,
      performFindAfterIndexingStarted: vi.fn(),
      reachablePreviewUrl: vi.fn(),
      stopFind: vi.fn()
    })

    expect(() => handles.get('hermes:openExternal')!(null, 'javascript:alert(1)')).toThrow('Invalid external URL')
  })
})
