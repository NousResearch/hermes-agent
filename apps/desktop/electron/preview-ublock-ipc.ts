import type { PreviewUblockPopupParent } from './preview-ublock-popup'
import type { PreviewUblockGuest, PreviewUblockRuntime } from './preview-ublock-runtime'

export interface PreviewUblockIpcSender extends PreviewUblockGuest {
  id: number
}

interface PreviewUblockIpcEvent {
  sender: PreviewUblockIpcSender
}

export interface PreviewUblockGuestWebContents extends PreviewUblockGuest {
  getType(): string
  hostWebContents: PreviewUblockIpcSender | null
  id: number
  isDestroyed(): boolean
}

interface PreviewUblockIpcMain {
  handle(channel: string, listener: (event: PreviewUblockIpcEvent, ...args: unknown[]) => unknown): void
}

export interface PreviewUblockBrowserWindow extends PreviewUblockPopupParent {}

export interface PreviewUblockIpcDependencies {
  browserWindowFromWebContents: (sender: PreviewUblockIpcSender) => PreviewUblockBrowserWindow | null
  getRuntime: () => PreviewUblockRuntime | null
  ipcMain: PreviewUblockIpcMain
  webContentsFromId: (webContentsId: number) => PreviewUblockGuestWebContents | null
}

function previewUblockGuestFor(
  event: PreviewUblockIpcEvent,
  webContentsFromId: PreviewUblockIpcDependencies['webContentsFromId'],
  webContentsId: unknown
): PreviewUblockGuestWebContents | null {
  if (!Number.isInteger(webContentsId)) {
    return null
  }

  const guest = webContentsFromId(webContentsId as number)

  if (!guest || guest.isDestroyed() || guest.getType() !== 'webview' || guest.hostWebContents !== event.sender) {
    return null
  }

  return guest
}

export function registerPreviewUblockIpc({
  browserWindowFromWebContents,
  getRuntime,
  ipcMain,
  webContentsFromId
}: PreviewUblockIpcDependencies): void {
  ipcMain.handle('hermes:preview-ublock:get', async () => {
    const runtime = getRuntime()

    if (!runtime) {
      throw new Error('uBlock Origin Lite is not ready')
    }

    return runtime.getState()
  })

  ipcMain.handle('hermes:preview-ublock:guest-register', (event, webContentsId) => {
    const guest = previewUblockGuestFor(event, webContentsFromId, webContentsId)
    const runtime = getRuntime()

    if (!guest || !runtime) {
      return { ok: false }
    }

    return { ok: runtime.registerGuest(guest.id, event.sender.id, guest) }
  })

  ipcMain.handle('hermes:preview-ublock:guest-unregister', (event, webContentsId) => {
    const guest = previewUblockGuestFor(event, webContentsFromId, webContentsId)
    const runtime = getRuntime()

    if (!guest || !runtime) {
      return { ok: false }
    }

    runtime.unregisterGuest(guest.id)

    return { ok: true }
  })

  ipcMain.handle('hermes:preview-ublock:open-popup', async event => {
    const parent = browserWindowFromWebContents(event.sender)
    const runtime = getRuntime()

    if (!parent || !runtime) {
      throw new Error('Preview uBlock popup could not be opened')
    }

    await runtime.openPopup(parent)

    return { ok: true }
  })

  ipcMain.handle('hermes:preview-ublock:set-enabled', async (_event, enabled) => {
    const runtime = getRuntime()

    if (!runtime) {
      throw new Error('uBlock Origin Lite is not ready')
    }

    return runtime.setEnabled(enabled === true)
  })
}
