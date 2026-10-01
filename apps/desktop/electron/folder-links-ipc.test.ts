import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, it, vi } from 'vitest'

const electron = vi.hoisted(() => ({
  bridge: {} as Record<string, (...args: unknown[]) => Promise<unknown>>,
  handlers: new Map<string, (event: { sender: { id: number } }, ...args: unknown[]) => unknown>(),
  senderId: 7,
  openPath: vi.fn(async (_path: string) => '')
}))

vi.mock('electron', () => ({
  contextBridge: {
    exposeInMainWorld: (_name: string, api: typeof electron.bridge) => {
      electron.bridge = api
    }
  },
  ipcRenderer: {
    sendSync: vi.fn(),
    invoke: (channel: string, ...args: unknown[]) =>
      electron.handlers.get(channel)?.({ sender: { id: electron.senderId } }, ...args)
  },
  ipcMain: {
    handle: (channel: string, handler: (event: { sender: { id: number } }, ...args: unknown[]) => unknown) =>
      electron.handlers.set(channel, handler)
  },
  shell: { openPath: electron.openPath },
  webFrame: {},
  webUtils: {}
}))

import './preload'

import { openExistingDirectory } from './folder-links'
import { registerFsIpc } from './fs-ipc'
import { WindowConnectionRouteRegistry } from './window-connection-route'

it('carries the preload request through filesystem IPC with the actual sender window, leaving openDir semantics intact', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-folder-ipc-'))

  try {
    const owner = { connectionId: 'local', profile: 'default' }
    const routes = new WindowConnectionRouteRegistry()
    routes.set(7, { ...owner, registryScoped: true })
    registerFsIpc({
      hermesHome: root,
      readActiveDesktopProfile: () => null,
      expandUserPath: p => p,
      resolveRequestedPathForIpc: p => p,
      directoryExists: p => fs.existsSync(p),
      resolveGitBinary: () => 'git',
      openExistingDirectory: (senderId, request) =>
        openExistingDirectory(senderId, request, {
          getRoute: id => routes.get(id),
          ensureBackend: async () => ({ mode: 'local' }),
          openPath: electron.openPath
        })
    })
    expect(electron.bridge.openExistingDirectory).toBeTypeOf('function')
    expect(await electron.bridge.openExistingDirectory({ path: root, owner })).toEqual({ ok: true })
    expect(electron.openPath).toHaveBeenCalledWith(await fs.promises.realpath(root))
    electron.senderId = 8
    electron.openPath.mockClear()
    expect(await electron.bridge.openExistingDirectory({ path: root, owner })).toMatchObject({ ok: false })
    expect(electron.openPath).not.toHaveBeenCalled()
    const missing = path.join(root, 'plugins')
    expect(await electron.bridge.openDir(missing)).toEqual({ ok: true })
    expect((await fs.promises.stat(missing)).isDirectory()).toBe(true)
  } finally {
    await fs.promises.rm(root, { recursive: true, force: true })
  }
})
