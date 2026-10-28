import { BrowserWindow, ipcMain, session, webContents } from 'electron'

import { notePreviewGuestHidden } from './preview-guest-offscreen'
import {
  beginPreviewGuestActivity,
  endPreviewGuestActivity,
  executePreviewGuestScript,
  PREVIEW_GUEST_PARTITION
} from './preview-guest-throttle'

export function registerPreviewGuestIpc(holdWindow: (owner: BrowserWindow) => () => void): void {
  // Kept-alive, off-screen pages must not receive focused-guest gestures.
  ipcMain.on('hermes:preview-guest-hidden', (event, payload) => {
    notePreviewGuestHidden(event.sender, webContents.fromId(Number(payload?.webContentsId)), Boolean(payload?.hidden))
  })

  ipcMain.handle('hermes:preview-guest-activity:begin', (event, webContentsId: unknown) => {
    if (!Number.isSafeInteger(webContentsId)) {
      throw new Error('Invalid preview guest')
    }

    const owner = BrowserWindow.fromWebContents(event.sender)

    if (!owner) {
      throw new Error('Preview owner window is unavailable')
    }

    return beginPreviewGuestActivity(
      event.sender,
      webContents.fromId(webContentsId as number) ?? undefined,
      session.fromPartition(PREVIEW_GUEST_PARTITION),
      () => holdWindow(owner)
    )
  })

  ipcMain.handle('hermes:preview-guest-activity:end', (event, token: unknown) => {
    if (Number.isSafeInteger(token)) {
      endPreviewGuestActivity(event.sender, token as number)
    }
  })

  ipcMain.handle('hermes:preview-guest-activity:script', (event, token: unknown, code: unknown) => {
    if (!Number.isSafeInteger(token) || typeof code !== 'string') {
      throw new Error('Invalid preview script request')
    }

    return executePreviewGuestScript(event.sender, token as number, code)
  })
}
