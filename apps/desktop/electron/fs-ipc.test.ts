/**
 * `hermes:fs:reveal` answers what it did (#115167). `shell.showItemInFolder`
 * selects an existing item and silently no-ops on a missing one, and a remote
 * backend's paths are missing on this computer by construction — a `true` for
 * them left the renderer nothing to say.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

const electron = vi.hoisted(() => ({
  handlers: new Map<string, (...args: unknown[]) => unknown>(),
  showItemInFolder: vi.fn()
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, handler: (...args: unknown[]) => unknown) => electron.handlers.set(channel, handler)
  },
  shell: {
    showItemInFolder: electron.showItemInFolder,
    openPath: vi.fn(async () => '')
  }
}))

vi.mock('./desktop-plugin-install', () => ({ installDesktopPluginFromGit: vi.fn(), probePluginRepo: vi.fn() }))
vi.mock('./desktop-plugins-root', () => ({
  DESKTOP_PLUGINS_DIR: 'desktop-plugins',
  ensureDir: vi.fn(),
  migrateProfileScopedDesktopPlugins: vi.fn(),
  reconcileUnifiedDesktopHalves: vi.fn()
}))

import { registerFsIpc } from './fs-ipc'

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-ipc-'))

registerFsIpc({
  hermesHome: scratch,
  readActiveDesktopProfile: () => null,
  // `~/` resolves under the scratch dir so tilde paths can be exercised.
  expandUserPath: value => (value.startsWith('~/') ? path.join(scratch, value.slice(2)) : value),
  resolveRequestedPathForIpc: value => value,
  directoryExists: value => fs.existsSync(value),
  resolveGitBinary: () => 'git'
})

const reveal = (target: string) => electron.handlers.get('hermes:fs:reveal')!({}, target)

afterEach(() => {
  electron.showItemInFolder.mockClear()
})

describe('hermes:fs:reveal', () => {
  it('reveals a path that exists on this computer', async () => {
    const file = path.join(scratch, 'workspace')
    fs.mkdirSync(file)

    await expect(reveal(file)).resolves.toBe(true)
    expect(electron.showItemInFolder).toHaveBeenCalledWith(file)
  })

  // A remote backend's workspace is not on this machine: showItemInFolder
  // would silently no-op, so the door must report the miss instead of success.
  it('reports false without touching the file manager when the path is missing', async () => {
    await expect(reveal(path.join(scratch, 'not-here'))).resolves.toBe(false)
    expect(electron.showItemInFolder).not.toHaveBeenCalled()
  })

  it('reports false for a path that is not on this computer, and only shows one that is', async () => {
    const here = path.join(scratch, 'notes.md')

    fs.writeFileSync(here, 'x')

    await expect(reveal('/home/hermes/.hermes/attachments/report.zip')).resolves.toBe(false)
    expect(electron.showItemInFolder).not.toHaveBeenCalled()

    await expect(reveal(here)).resolves.toBe(true)
    expect(electron.showItemInFolder).toHaveBeenCalledTimes(1)
    expect(electron.showItemInFolder).toHaveBeenCalledWith(here)
  })

  // The renderer may hand over a tilde path; the existence check runs on the
  // expanded path, and the expanded path is what the file manager is shown.
  it('expands a tilde path before checking and revealing it', async () => {
    const here = path.join(scratch, 'tilde.md')

    fs.writeFileSync(here, 'x')

    await expect(reveal('~/tilde.md')).resolves.toBe(true)
    expect(electron.showItemInFolder).toHaveBeenCalledWith(here)
  })
})
