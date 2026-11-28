/**
 * `hermes:fs:desktopPluginsRoot` builds the app's plugin root from the profile homes. A home a sandbox can
 * write must not feed it: the handler asks the check `registerFsIpc` is given, for every profile.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, expect, it, vi } from 'vitest'

const electron = vi.hoisted(() => ({ handlers: new Map<string, (...args: unknown[]) => unknown>() }))

vi.mock('electron', () => ({
  app: { getPath: () => os.tmpdir() },
  BrowserWindow: { fromWebContents: () => null },
  dialog: { showOpenDialog: vi.fn() },
  ipcMain: {
    handle: (channel: string, handler: (...args: unknown[]) => unknown) => electron.handlers.set(channel, handler)
  },
  shell: { openPath: vi.fn(async () => ''), showItemInFolder: vi.fn() }
}))

vi.mock('./desktop-plugin-install', () => ({ installDesktopPluginFromGit: vi.fn(), probePluginRepo: vi.fn() }))

import { registerFsIpc } from './fs-ipc'

const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-ipc-root-'))

afterAll(() => fs.rmSync(home, { force: true, recursive: true }))

function write(file: string, text: string) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, text)
}

it('the plugin root takes nothing from a home the sandbox check names', async () => {
  write(path.join(home, 'profiles', 'work', 'plugins', 'planted', 'desktop', 'plugin.js'), 'export default {}')
  write(path.join(home, 'profiles', 'work', 'desktop-plugins', 'scoped', 'plugin.js'), 'export default {}')
  write(path.join(home, 'profiles', 'other', 'plugins', 'fine', 'desktop', 'plugin.js'), 'export default {}')
  write(path.join(home, 'profiles', 'other', 'desktop-plugins', 'moved', 'plugin.js'), 'export default {}')

  registerFsIpc({
    directoryExists: value => fs.existsSync(value),
    expandUserPath: value => value,
    hermesHome: home,
    isLaunchedProfile: profile => profile === 'work',
    readActiveDesktopProfile: () => null,
    resolveGitBinary: () => 'git',
    resolveRequestedPathForIpc: value => value
  })

  const root = String(await electron.handlers.get('hermes:fs:desktopPluginsRoot')!({}))

  expect(root).toBe(path.join(home, 'desktop-plugins'))
  expect(fs.existsSync(path.join(root, 'fine', 'plugin.js'))).toBe(true)
  expect(fs.existsSync(path.join(root, 'moved', 'plugin.js'))).toBe(true)
  expect(fs.existsSync(path.join(root, 'planted'))).toBe(false)
  expect(fs.existsSync(path.join(root, 'scoped'))).toBe(false)
  // The sandboxed home keeps its own files: nothing was moved out of it.
  expect(fs.existsSync(path.join(home, 'profiles', 'work', 'desktop-plugins', 'scoped', 'plugin.js'))).toBe(true)
})
