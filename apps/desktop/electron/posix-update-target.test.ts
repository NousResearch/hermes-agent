import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as updaterProcess from './updater-process'

const host = vi.hoisted(() => ({
  appRoot: '',
  userData: '',
  home: '',
  handlers: new Map<string, (...args: any[]) => any>(),
  spawnUpdater: vi.fn((_command: string, _args: string[], _options: unknown) => ({})),
  quit: vi.fn()
}))

// Register the real main-process IPC without starting Electron, a backend,
// or an updater. Git and the saved update configuration remain real.
vi.mock('electron', () => ({
  app: {
    isPackaged: false,
    getAppPath: () => host.appRoot,
    getPath: (name: string) => (name === 'userData' ? host.userData : host.home),
    setPath: vi.fn(),
    getVersion: () => '0.0.0',
    setName: vi.fn(),
    setAboutPanelOptions: vi.fn(),
    commandLine: { appendSwitch: vi.fn() },
    disableHardwareAcceleration: vi.fn(),
    requestSingleInstanceLock: () => true,
    on: vi.fn(),
    whenReady: () => new Promise(() => {}),
    quit: host.quit
  },
  ipcMain: {
    handle: (channel: string, handler: (...args: any[]) => any) => host.handlers.set(channel, handler),
    on: vi.fn()
  },
  BrowserWindow: { getAllWindows: () => [] },
  clipboard: {},
  dialog: {},
  net: {},
  webContents: {},
  globalShortcut: {},
  Menu: {},
  nativeTheme: {},
  Notification: {},
  powerMonitor: {},
  powerSaveBlocker: {},
  protocol: { registerSchemesAsPrivileged: vi.fn() },
  safeStorage: {},
  screen: {},
  session: {},
  shell: {},
  systemPreferences: {}
}))

vi.mock('./crash-forensics', () => ({ installCrashForensics: vi.fn() }))
vi.mock('./updater-process', async importOriginal => ({
  ...(await importOriginal<typeof updaterProcess>()),
  spawnUpdaterProcess: host.spawnUpdater,
  observeUpdaterHandoff: vi.fn(async () => ({ ok: true }))
}))

describe.skipIf(process.platform === 'win32')('POSIX update target', () => {
  let sandbox: string
  let checkout: string
  let remote: string

  const git = (cwd: string, ...args: string[]) =>
    execFileSync('git', args, {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    }).trim()

  const invoke = (channel: string, ...args: unknown[]) => {
    const handler = host.handlers.get(channel)
    expect(handler, `IPC handler ${channel}`).toBeTypeOf('function')

    return handler!({}, ...args)
  }

  beforeAll(async () => {
    sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-posix-update-target-'))
    host.home = path.join(sandbox, 'home')
    host.userData = path.join(sandbox, 'user-data')
    checkout = path.join(host.home, 'hermes-agent')
    host.appRoot = path.join(checkout, 'apps', 'desktop')
    remote = path.join(sandbox, 'origin.git')
    fs.mkdirSync(host.appRoot, { recursive: true })
    vi.stubEnv('HERMES_HOME', host.home)
    vi.stubEnv('HERMES_DESKTOP_USER_DATA_DIR', host.userData)
    vi.stubEnv('HERMES_DESKTOP_HERMES_ROOT', checkout)
    vi.stubEnv('HERMES_DESKTOP_CDP_PORT', 'off')
    vi.stubEnv('GIT_CONFIG_GLOBAL', path.join(sandbox, 'no-global-config'))
    vi.stubEnv('GIT_CONFIG_NOSYSTEM', '1')
    git(checkout, 'init', '--initial-branch=main')
    git(
      checkout,
      '-c',
      'user.name=Updater Test',
      '-c',
      'user.email=updater@example.invalid',
      'commit',
      '--allow-empty',
      '-m',
      'base'
    )
    git(checkout, 'branch', 'maintenance/reports')
    git(checkout, 'branch', 'feature/desktop')
    git(checkout, 'branch', 'feature/deleted')
    git(checkout, 'checkout', 'maintenance/reports')
    git(
      checkout,
      '-c',
      'user.name=Updater Test',
      '-c',
      'user.email=updater@example.invalid',
      'commit',
      '--allow-empty',
      '-m',
      'maintained report changes'
    )
    git(sandbox, 'clone', '--bare', checkout, remote)
    git(remote, 'update-ref', '-d', 'refs/heads/feature/deleted')
    git(checkout, 'remote', 'add', 'origin', remote)
    const script = path.join(checkout, 'scripts', 'desktop-update', 'posix.sh')
    fs.mkdirSync(path.dirname(script), { recursive: true })
    fs.writeFileSync(script, '#!/bin/bash\nexit 93\n')

    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    await import('./main')
  })

  beforeEach(() => {
    host.spawnUpdater.mockClear()
    host.quit.mockClear()
  })

  afterAll(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.unstubAllEnvs()

    if (sandbox) {
      fs.rmSync(sandbox, { recursive: true, force: true })
    }
  })

  it.each([
    { configured: 'main', expected: 'main' },
    { configured: 'feature/desktop', expected: 'feature/desktop' },
    { configured: 'feature/deleted', expected: 'main' }
  ])('hands off saved $configured as $expected while maintenance is published', async ({ configured, expected }) => {
    expect(git(checkout, 'branch', '--show-current')).toBe('maintenance/reports')
    expect(git(checkout, 'ls-remote', '--exit-code', '--heads', 'origin', 'maintenance/reports')).not.toBe('')
    const head = git(checkout, 'rev-parse', 'HEAD')
    await invoke('hermes:updates:branch:set', configured)

    expect(await invoke('hermes:updates:apply')).toMatchObject({ ok: true, handedOff: true })
    expect(host.spawnUpdater).toHaveBeenCalledOnce()
    expect(host.spawnUpdater).toHaveBeenCalledWith(
      '/bin/bash',
      expect.arrayContaining([
        path.join(checkout, 'scripts', 'desktop-update', 'posix.sh'),
        '--install-root',
        checkout,
        '--branch',
        expected,
        '--desktop-pid',
        String(process.pid)
      ]),
      expect.objectContaining({ cwd: host.home, detached: true, stdio: 'ignore' })
    )
    const args = host.spawnUpdater.mock.calls[0][1] as string[]
    expect(args[args.indexOf('--branch') + 1]).toBe(expected)
    expect(await invoke('hermes:updates:branch:get')).toEqual({ branch: expected })
    expect(JSON.parse(fs.readFileSync(path.join(host.userData, 'updates.json'), 'utf8'))).toEqual({ branch: expected })
    expect(git(checkout, 'branch', '--show-current')).toBe('maintenance/reports')
    expect(git(checkout, 'rev-parse', 'HEAD')).toBe(head)
    expect(host.quit).not.toHaveBeenCalled()
  })
})
