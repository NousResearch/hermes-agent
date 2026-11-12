// Clipboard text IPC for the renderer. Electron 44's clipboard calls follow the async W3C
// clipboard API, so each handler settles only after the OS clipboard call does.

import type { Clipboard, IpcMain, IpcMainInvokeEvent } from 'electron'

export function registerClipboardTextIpc(
  ipcMain: Pick<IpcMain, 'handle'>,
  clipboard: Pick<Clipboard, 'readText' | 'writeText'>
): void {
  ipcMain.handle('hermes:writeClipboard', async (_event: IpcMainInvokeEvent, text: unknown): Promise<boolean> => {
    await clipboard.writeText(String(text || ''))

    return true
  })

  // Paired reader for the GUI terminal's paste chord: the renderer's
  // navigator.clipboard.readText() throws "Document is not focused" whenever a
  // portaled overlay has focus, and there's no way to route a read through the
  // canvas. The main process has no such gate.
  ipcMain.handle('hermes:readClipboard', (): Promise<string> => clipboard.readText())
}
