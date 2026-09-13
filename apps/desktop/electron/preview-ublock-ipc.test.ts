import { describe, expect, it, vi } from 'vitest'

import {
  type PreviewUblockBrowserWindow,
  type PreviewUblockGuestWebContents,
  type PreviewUblockIpcSender,
  registerPreviewUblockIpc
} from './preview-ublock-ipc'
import type { PreviewUblockRuntime } from './preview-ublock-runtime'

function sender(id = 7): PreviewUblockIpcSender {
  return {
    id,
    on: vi.fn(),
    removeListener: vi.fn()
  }
}

function guest(owner: PreviewUblockIpcSender, id = 12, type = 'webview'): PreviewUblockGuestWebContents {
  return {
    getType: () => type,
    hostWebContents: owner,
    id,
    isDestroyed: () => false,
    on: vi.fn(),
    removeListener: vi.fn()
  }
}

function runtime(): PreviewUblockRuntime {
  return {
    dispose: vi.fn().mockResolvedValue(undefined),
    getBlockedRequestCount: vi.fn().mockReturnValue(0),
    getRequestBlockerBlockedRequestCount: vi.fn().mockReturnValue(0),
    getState: vi.fn().mockReturnValue({}),
    initialize: vi.fn().mockResolvedValue({}),
    loadRules: vi.fn().mockReturnValue(true),
    openPopup: vi.fn().mockResolvedValue(undefined),
    recordBlockedRequest: vi.fn(),
    registerGuest: vi.fn().mockReturnValue(true),
    setEnabled: vi.fn().mockResolvedValue({}),
    unregisterGuest: vi.fn()
  }
}

describe('preview uBlock IPC', () => {
  it('rejects invalid or foreign guests before reaching the runtime', async () => {
    const handlers = new Map<string, (...args: unknown[]) => unknown>()
    const owner = sender()
    const runtimeValue = runtime()
    const guests = new Map<number, PreviewUblockGuestWebContents>()
    const parent = {} as PreviewUblockBrowserWindow
    registerPreviewUblockIpc({
      browserWindowFromWebContents: () => parent,
      getRuntime: () => runtimeValue,
      ipcMain: { handle: (channel, handler) => handlers.set(channel, handler) },
      webContentsFromId: id => guests.get(id) ?? null
    })

    const register = handlers.get('hermes:preview-ublock:guest-register')!
    const foreign = sender(8)
    guests.set(12, guest(foreign))

    expect(register({ sender: owner }, '12')).toEqual({ ok: false })
    expect(register({ sender: owner }, 12)).toEqual({ ok: false })
    expect(runtimeValue.registerGuest).not.toHaveBeenCalled()
  })

  it('registers and unregisters a live guest owned by the requesting renderer', async () => {
    const handlers = new Map<string, (...args: unknown[]) => unknown>()
    const owner = sender()
    const runtimeValue = runtime()
    const previewGuest = guest(owner)
    registerPreviewUblockIpc({
      browserWindowFromWebContents: () => ({}) as PreviewUblockBrowserWindow,
      getRuntime: () => runtimeValue,
      ipcMain: { handle: (channel, handler) => handlers.set(channel, handler) },
      webContentsFromId: id => (id === previewGuest.id ? previewGuest : null)
    })

    const event = { sender: owner }
    expect(handlers.get('hermes:preview-ublock:guest-register')!(event, previewGuest.id)).toEqual({ ok: true })
    expect(handlers.get('hermes:preview-ublock:guest-unregister')!(event, previewGuest.id)).toEqual({ ok: true })
    expect(runtimeValue.registerGuest).toHaveBeenCalledWith(previewGuest.id, owner.id, previewGuest)
    expect(runtimeValue.unregisterGuest).toHaveBeenCalledWith(previewGuest.id)
  })
})
