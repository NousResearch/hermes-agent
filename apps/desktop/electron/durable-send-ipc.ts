import { randomUUID } from 'node:crypto'

import type { BrowserWindow, IpcMain, IpcMainInvokeEvent, WebContents } from 'electron'

import { drainDurableSends, DurableSendRefusal } from './durable-send-delivery'
import { DURABLE_RPC_METHODS, type DurableRpc, durableRpcTransport } from './durable-send-rpc'
import { DurableSendStore } from './durable-send-store'
import type { DurableSendOwner } from './durable-send-types'

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

  // Use the application's existing authenticated owner-scoped sockets, not a second credential
  // implementation. Replies are single-use, timeout-bounded and bound to the requesting app frame.
  const pending = new Map<
    string,
    {
      sender: WebContents
      resolve: (value: any) => void
      reject: (error: Error) => void
      timer: ReturnType<typeof setTimeout>
    }
  >()

  const rpcFor = (event: IpcMainInvokeEvent): DurableRpc => {
    let leaseId: string | undefined

    const rpc: DurableRpc = (route, method, params) => {
      store(event)

      if (!DURABLE_RPC_METHODS.has(method) && !['transport.acquire', 'transport.release'].includes(method)) {
        return Promise.reject(new DurableSendRefusal('RPC denied'))
      }

      return new Promise((resolve, reject) => {
        const token = randomUUID()

        const timer = setTimeout(() => {
          pending.delete(token)
          reject(new Error('Transport unavailable'))
        }, 30_000)

        pending.set(token, { sender: event.sender, resolve, reject, timer })
        event.sender.send('hermes:durable-send:transport', { token, route, method, params, leaseId })
      })
    }

    rpc.acquire = async owner => {
      leaseId = randomUUID()

      try {
        await rpc(owner, 'transport.acquire', {})
      } catch (error) {
        await rpc(owner, 'transport.release', {}).catch(() => {})
        leaseId = undefined
        throw error
      }

      return async () => {
        try { await rpc(owner, 'transport.release', {}) } finally { leaseId = undefined }
      }
    }

    return rpc
  }

  ipc.handle('hermes:durable-send:transport-reply' , (event, token: string, reply: any) => {
    store(event)
    const request = pending.get(token)

    if (!request || request.sender !== event.sender) {
      throw new Error('Transport reply denied')
    }

    pending.delete(token)
    clearTimeout(request.timer)

    if (reply?.ok === true) {
      request.resolve(reply.result)
    } else {
      request.reject(
        reply?.refused === true ? new DurableSendRefusal('Gateway refused') : new Error('Transport unavailable')
      )
    }
  })
  ipc.handle(
    'hermes:durable-send:negotiate',
    async (event, route: Pick<DurableSendOwner, 'connectionId' | 'profile'>) => {
      store(event)

      if (
        !route ||
        typeof route.connectionId !== 'string' ||
        !route.connectionId ||
        typeof route.profile !== 'string' ||
        !route.profile
      ) {
        throw new Error('Missing exact owner')
      }

      const result = await durableRpcTransport(rpcFor(event)).describe({ ...route, gatewayId: '', principalId: '' })

      if (!result.uploadReplayVerified) {
        throw new Error('Durable send capability unavailable')
      }

      return result.owner
    }
  )
  ipc.handle('hermes:durable-send:pump', event => {
    const transport = durableRpcTransport(rpcFor(event))
    transport.progress = () => event.sender.send('hermes:durable-send:changed')

    return drainDurableSends(store(event), transport)
  })

  ipc.handle('hermes:durable-send:begin', (event, intent: unknown) => store(event).begin(intent))
  ipc.handle('hermes:durable-send:append', (event, id: string, fileId: string, offset: number, bytes: Uint8Array) =>
    store(event).append(id, fileId, offset, bytes)
  )
  ipc.handle('hermes:durable-send:commit', (event, id: string) => store(event).commit(id))
  ipc.handle('hermes:durable-send:list', event => store(event).list())
  ipc.handle('hermes:durable-send:retired', (event, ids: string[]) => {
    const spool = store(event)

    if (!Array.isArray(ids) || ids.length > 128) {throw new Error('Invalid receipt query')}

    return Object.fromEntries(ids.map(id => [id, spool.retiredState(id)]).filter(([, state]) => state))
  })
  ipc.handle('hermes:durable-send:cancel', (event, id: string) => store(event).cancel(id))
}
