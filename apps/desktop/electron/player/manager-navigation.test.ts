import { EventEmitter } from 'node:events'

import type { BrowserWindow, Session, WebContentsView } from 'electron'
import { describe, expect, it, vi } from 'vitest'

import { PlayerManager } from './manager'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, resolve, reject }
}

describe('PlayerManager navigation scheduling', () => {
  it('keeps bounds and subsequent navigation commands responsive while a page load is pending', async () => {
    const fixtureUrl = 'http://127.0.0.1:9876/player'
    const initialLoad = deferred<void>()
    const homeLoad = deferred<void>()
    const loads = [initialLoad, homeLoad]

    const webContents = Object.assign(new EventEmitter(), {
      id: 41,
      setWindowOpenHandler: vi.fn(),
      loadURL: vi.fn(() => loads.shift()!.promise),
      isDestroyed: vi.fn(() => false),
      getURL: vi.fn(() => fixtureUrl),
      canGoBack: vi.fn(() => false),
      canGoForward: vi.fn(() => false),
      close: vi.fn()
    })

    const guest = {
      webContents,
      setBounds: vi.fn()
    } as unknown as WebContentsView

    const ownerWebContents = new EventEmitter()

    const owner = Object.assign(new EventEmitter(), {
      id: 7,
      isDestroyed: vi.fn(() => false),
      contentView: { addChildView: vi.fn(), removeChildView: vi.fn() },
      webContents: ownerWebContents,
      getContentSize: vi.fn(() => [1280, 720] as [number, number])
    }) as unknown as BrowserWindow

    const ses = {
      setPermissionRequestHandler: vi.fn(),
      setPermissionCheckHandler: vi.fn()
    } as unknown as Session

    const manager = new PlayerManager({
      allowFixture: true,
      fixtureUrl,
      createView: () => guest,
      getSession: () => ses
    })

    let opening: Promise<unknown> | null = null

    try {
      const open = manager.request(owner, { type: 'open', source: 'youtube' })
      opening = open

      const opened = await Promise.race([
        open.then(() => true),
        new Promise<boolean>(resolve => setTimeout(() => resolve(false), 100))
      ])

      expect(opened).toBe(true)
      expect(webContents.loadURL).toHaveBeenCalledWith(fixtureUrl)

      const bounds = manager.request(owner, {
        type: 'bounds',
        bounds: { x: 20, y: 30, width: 320, height: 180 },
        visible: true,
        revision: 1
      })

      const boundsApplied = await Promise.race([
        bounds.then(() => true),
        new Promise<boolean>(resolve => setTimeout(() => resolve(false), 100))
      ])

      expect(boundsApplied).toBe(true)
      expect(guest.setBounds).toHaveBeenCalledTimes(2)
      expect(guest.setBounds).toHaveBeenLastCalledWith({ x: 20, y: 30, width: 320, height: 180 })

      await manager.request(owner, { type: 'home' })
      expect(webContents.loadURL).toHaveBeenCalledTimes(2)

      webContents.emit('did-fail-load', {}, -3, 'ERR_ABORTED', fixtureUrl, true)
      const state = await manager.request(owner, { type: 'status' })
      expect(state.status).toBe('loading')
      expect(state.error).toBeNull()

      webContents.emit('did-fail-load', {}, -105, 'ERR_NAME_NOT_RESOLVED', fixtureUrl, true)
      const failed = await manager.request(owner, { type: 'status' })
      expect(failed.status).toBe('error')
      expect(failed.error).toBe('navigation-failed--105')
    } finally {
      manager.dispose()
      initialLoad.reject(new Error('late initial navigation failure'))
      homeLoad.reject(new Error('late home navigation failure'))
      await Promise.allSettled([initialLoad.promise, homeLoad.promise])
      await opening?.catch(() => undefined)
    }
  })
})
