/**
 * `hermes:fs:reveal` answers what it did (#115167). `shell.showItemInFolder`
 * selects an existing item and silently no-ops on a missing one, and a remote
 * backend's paths are missing on this computer by construction — a `true` for
 * them left the renderer nothing to say.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, afterEach, describe, expect, it, vi } from 'vitest'

const electron = vi.hoisted(() => ({
  handlers: new Map<string, (...args: unknown[]) => unknown>(),
  showItemInFolder: vi.fn()
}))

vi.mock('electron', () => ({
  app: { getPath: () => os.tmpdir() },
  BrowserWindow: { fromWebContents: () => null },
  dialog: { showOpenDialog: vi.fn() },
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
import { resolveRequestedPathForIpc } from './hardening'

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-ipc-'))

registerFsIpc({
  hermesHome: scratch,
  readActiveDesktopProfile: () => 'launch-profile',
  // `~/` resolves under the scratch dir so tilde paths can be exercised.
  expandUserPath: value => (value.startsWith('~/') ? path.join(scratch, value.slice(2)) : value),
  resolveRequestedPathForIpc,
  directoryExists: value => fs.existsSync(value),
  resolveGitBinary: () => 'git'
})

const reveal = (target: string) => electron.handlers.get('hermes:fs:reveal')!({}, target)

afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }))

describe('hermes:fs:createDirectoryExclusive', () => {
  const create = (target: unknown) => electron.handlers.get('hermes:fs:createDirectoryExclusive')!({}, target)

  it('creates exactly one directory, including tilde and file URL paths', async () => {
    const target = path.join(scratch, 'new folder')
    await expect(create('~/new folder')).resolves.toEqual({ path: target })
    expect(fs.statSync(target).isDirectory()).toBe(true)
    const { pathToFileURL } = await import('node:url')
    const urlTarget = path.join(scratch, 'url folder')
    await expect(create(pathToFileURL(urlTarget).href)).resolves.toEqual({ path: urlTarget })
  })

  it('never replaces existing directories, files or symlinks, and never creates parents', async () => {
    const dir = path.join(scratch, 'existing-dir')
    const file = path.join(scratch, 'existing-file')
    fs.mkdirSync(dir)
    fs.writeFileSync(file, 'keep')

    for (const target of [dir, file]) {await expect(create(target)).rejects.toThrow(/already exists/)}
    expect(fs.readFileSync(file, 'utf8')).toBe('keep')
    await expect(create(path.join(scratch, 'missing', 'child'))).rejects.toThrow()
    expect(fs.existsSync(path.join(scratch, 'missing'))).toBe(false)

    if (process.platform !== 'win32') {
      const link = path.join(scratch, 'dangling-link')
      fs.symlinkSync(path.join(scratch, 'nowhere'), link)
      await expect(create(link)).rejects.toThrow(/already exists/)
      expect(fs.lstatSync(link).isSymbolicLink()).toBe(true)
    }
  })

  it('rejects invalid inputs and device paths without creating anything', async () => {
    for (const target of ['', '  ', null, '\\\\?\\C:\\secret']) {
      await expect(create(target)).rejects.toThrow()
    }
  })

  it('propagates filesystem errors other than collisions', async () => {
    const failure = Object.assign(new Error('permission denied'), { code: 'EACCES' })
    const mkdir = vi.spyOn(fs.promises, 'mkdir').mockRejectedValueOnce(failure)

    try {
      await expect(create(path.join(scratch, 'denied'))).rejects.toBe(failure)
    } finally {
      mkdir.mockRestore()
    }
  })

  it('lets only one concurrent creator succeed', async () => {
    const target = path.join(scratch, 'raced')
    const results = await Promise.allSettled([create(target), create(target)])
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1)
    expect(results.filter(result => result.status === 'rejected')).toHaveLength(1)
  })
})

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
