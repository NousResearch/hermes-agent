/**
 * A folder grant is minted only by the native picker main opens itself: the
 * renderer names the plugin and the picker labels, never a path.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerPluginFilesIpc } from './plugin-files-ipc'

const scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-plugin-files-ipc-')))
const hermesHome = path.join(scratch, '.hermes')
const vault = path.join(scratch, 'Vault')
fs.mkdirSync(hermesHome)
fs.mkdirSync(vault)
fs.writeFileSync(path.join(vault, 'note.md'), 'hello')

const handlers = new Map<string, (...args: unknown[]) => Promise<unknown>>()
const showOpenDialog = vi.fn()
const broadcastChange = vi.fn()
const window = { id: 1 }

registerPluginFilesIpc({
  ipcMain: { handle: (channel: string, handler: never) => handlers.set(channel, handler) } as never,
  dialog: { showOpenDialog } as never,
  windowFor: () => window as never,
  broadcastChange,
  userDataDir: () => path.join(scratch, 'userData'),
  hermesHomes: () => [hermesHome],
  trashItem: async () => undefined
})

const call = (channel: string, ...args: unknown[]) => handlers.get(channel)!({ sender: {} }, ...args)

afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }))

beforeEach(() => broadcastChange.mockClear())

describe('hermes:pluginFiles:request', () => {
  it('a cancelled picker grants nothing', async () => {
    showOpenDialog.mockResolvedValueOnce({ canceled: true, filePaths: [] })

    expect(await call('hermes:pluginFiles:request', 'plexus', { title: 'Choose a folder for Plexus' })).toEqual({
      ok: true,
      folder: null
    })
    expect(showOpenDialog).toHaveBeenLastCalledWith(
      window,
      expect.objectContaining({ properties: ['openDirectory', 'createDirectory'], title: 'Choose a folder for Plexus' })
    )
    expect(await call('hermes:pluginFiles:grants', 'plexus')).toEqual([])
  })

  it('the Hermes home is refused even when the user picks it', async () => {
    showOpenDialog.mockResolvedValueOnce({ canceled: false, filePaths: [hermesHome] })

    expect(await call('hermes:pluginFiles:request', 'plexus')).toMatchObject({ ok: false, error: 'protected' })
    expect(await call('hermes:pluginFiles:grants', 'plexus')).toEqual([])
  })

  it('the picked folder becomes a grant the plugin can read through, until revoked', async () => {
    showOpenDialog.mockResolvedValueOnce({ canceled: false, filePaths: [vault] })

    const granted = (await call('hermes:pluginFiles:request', 'plexus')) as { folder: { id: string; path: string } }

    expect(granted.folder.path).toBe(vault)
    expect(
      await call('hermes:pluginFiles:op', 'plexus', { op: 'readText', folder: granted.folder.id, path: 'note.md' })
    ).toMatchObject({ ok: true, text: 'hello' })

    await call('hermes:pluginFiles:revoke', 'plexus', granted.folder.id)

    expect(
      await call('hermes:pluginFiles:op', 'plexus', { op: 'readText', folder: granted.folder.id, path: 'note.md' })
    ).toMatchObject({ ok: false, error: 'not-granted' })
  })

  it('a refused or cancelled pick tells no window anything', async () => {
    showOpenDialog.mockResolvedValueOnce({ canceled: true, filePaths: [] })
    await call('hermes:pluginFiles:request', 'plexus')
    showOpenDialog.mockResolvedValueOnce({ canceled: false, filePaths: [hermesHome] })
    await call('hermes:pluginFiles:request', 'plexus')

    expect(broadcastChange).not.toHaveBeenCalled()
  })
})

describe('every window hears about grant changes', () => {
  it('a grant and a revoke each broadcast the plugin id; a revoke of nothing does not', async () => {
    showOpenDialog.mockResolvedValueOnce({ canceled: false, filePaths: [vault] })
    const granted = (await call('hermes:pluginFiles:request', 'plexus')) as { folder: { id: string } }

    expect(broadcastChange).toHaveBeenCalledTimes(1)
    expect(broadcastChange).toHaveBeenLastCalledWith({ pluginId: 'plexus' })

    await call('hermes:pluginFiles:revoke', 'plexus', granted.folder.id)

    expect(broadcastChange).toHaveBeenCalledTimes(2)

    await call('hermes:pluginFiles:revoke', 'plexus', granted.folder.id)

    expect(broadcastChange).toHaveBeenCalledTimes(2)
  })
})
