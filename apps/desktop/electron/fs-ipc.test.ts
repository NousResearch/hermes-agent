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
  ensureDir: vi.fn(async (dir: string) => dir),
  migrateProfileScopedDesktopPlugins: vi.fn(),
  reconcileUnifiedDesktopHalves: vi.fn()
}))

import { registerFsIpc } from './fs-ipc'

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-ipc-'))

registerFsIpc({
  hermesHome: scratch,
  readActiveDesktopProfile: () => 'launch-profile',
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

  // The renderer may hand over a tilde path; the existence check runs on the
  // expanded path, and the expanded path is what the file manager is shown.
  it('expands a tilde path before checking and revealing it', async () => {
    const here = path.join(scratch, 'tilde.md')

    fs.writeFileSync(here, 'x')

    await expect(reveal('~/tilde.md')).resolves.toBe(true)
    expect(electron.showItemInFolder).toHaveBeenCalledWith(here)
  })
})

// A pooled backend serves several profile homes; the active Desktop profile
// is the LAUNCH one, so the error card names the profile owning the failing
// session and the root resolves under THAT home (#119080).
describe('hermes:fs:logsRoot', () => {
  const logsRoot = (profile?: string) => electron.handlers.get('hermes:fs:logsRoot')!({}, profile)

  it('resolves the logs dir of the profile that owns the session', async () => {
    await expect(logsRoot('finex')).resolves.toBe(path.join(scratch, 'profiles', 'finex', 'logs'))
    await expect(logsRoot('default')).resolves.toBe(path.join(scratch, 'logs'))
  })

  it('falls back to the active Desktop profile when no owner is named', async () => {
    await expect(logsRoot()).resolves.toBe(path.join(scratch, 'profiles', 'launch-profile', 'logs'))
  })

  // The owner is renderer data (from a remote backend in remote mode): a
  // traversal or absolute value must never leave hermesHome, let alone be
  // created and revealed. Bad names route like an unnamed owner.
  it('never leaves hermesHome for an owner that is not a profile name', async () => {
    for (const bad of ['../../x', '/etc', 'a/b', 'Upper', '.hidden']) {
      const root = (await logsRoot(bad)) as string

      expect(path.relative(scratch, root).startsWith('..')).toBe(false)
      expect(root).toBe(path.join(scratch, 'profiles', 'launch-profile', 'logs'))
    }
  })
})
