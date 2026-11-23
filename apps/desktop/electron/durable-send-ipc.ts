import type { BrowserWindow, IpcMain, IpcMainInvokeEvent, WebContents } from 'electron'

import { DurableSendStore } from './durable-send-store'

interface Dependencies {
  ipc: Pick<IpcMain, 'handle'>
  userData: () => string
  rendererUrl: () => string
  getOwner: (sender: WebContents) => BrowserWindow | null
}

/** Only the actual app document in an owning top-level window may use the spool.
 * Guest frames and a navigated external document are refused.
 * Renderer arguments are identities/chunks, never filesystem paths or terminal receipts. */
export function registerDurableSendIpc({ ipc, userData, rendererUrl, getOwner }: Dependencies): void {
  const stores = new Map<string, DurableSendStore>()

  const store = (event: IpcMainInvokeEvent) => {
    const owner = getOwner(event.sender)

    if (
      !owner ||
      owner.isDestroyed() ||
      owner.webContents !== event.sender ||
      event.senderFrame !== event.sender.mainFrame
    ) {
      throw new Error('Durable send IPC denied')
    }

    const actual = new URL(event.senderFrame.url)
    const expected = new URL(rendererUrl())

    if (
      actual.protocol !== expected.protocol ||
      actual.origin !== expected.origin ||
      actual.pathname !== expected.pathname
    ) {
      throw new Error('Durable send IPC denied')
    }

    const root = userData()
    const key = JSON.stringify([root, expected.origin, expected.pathname])
    let result = stores.get(key)

    if (!result) {
      result = new DurableSendStore(root, key)
      stores.set(key, result)
    }

    return result
  }

  ipc.handle('hermes:durable-send:begin', (event, intent: unknown) => store(event).begin(intent))
  ipc.handle('hermes:durable-send:append', (event, id: string, fileId: string, offset: number, bytes: Uint8Array) =>
    store(event).append(id, fileId, offset, bytes)
  )
  ipc.handle('hermes:durable-send:commit', (event, id: string) => store(event).commit(id))
  ipc.handle('hermes:durable-send:list', event => store(event).list())
  ipc.handle('hermes:durable-send:cancel', (event, id: string) => store(event).cancel(id))
}
