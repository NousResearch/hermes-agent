import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  classifyMacBundleLocation,
  guardMacApplicationsLaunch,
  MAC_APPLICATIONS_GUARD_FILENAME,
  type MacApplicationsGuardDeps,
  macApplicationsGuardPreferencePath,
  readMacApplicationsGuardPreference,
  resolveMacAppBundle
} from './mac-applications-guard'

const tempDirs: string[] = []

function makeTempUserData(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mac-apps-guard-'))

  tempDirs.push(dir)

  return dir
}

function makeDeps(overrides: Partial<MacApplicationsGuardDeps> = {}) {
  const userDataDir = makeTempUserData()
  const logs: string[] = []

  const getPath = vi.fn((name: string) => {
    if (name === 'exe') {
      return '/Volumes/Hermes Agent/Hermes Agent.app/Contents/MacOS/Hermes Agent'
    }

    if (name === 'userData') {
      return userDataDir
    }

    return userDataDir
  })

  const app = {
    getPath: getPath as unknown as MacApplicationsGuardDeps['app']['getPath'],
    getVersion: vi.fn(() => '0.24.0'),
    isInApplicationsFolder: vi.fn(() => false),
    moveToApplicationsFolder: vi.fn((_options?: Electron.MoveToApplicationsFolderOptions) => true)
  }

  const dialog = {
    showMessageBox: vi.fn(async () => ({ response: 0, checkboxChecked: false }))
  }

  const access = vi.fn(() => {
    throw Object.assign(new Error('read-only file system'), { code: 'EROFS' })
  })

  const deps: MacApplicationsGuardDeps = {
    app,
    dialog,
    appName: 'Hermes Agent',
    isPackaged: true,
    installStamp: { payload: 'bundled', updateMechanism: 'electron-updater' },
    platform: 'darwin',
    env: {},
    argv: ['/Volumes/Hermes Agent/Hermes Agent.app/Contents/MacOS/Hermes Agent'],
    homeDir: '/Users/alice',
    access,
    log: message => {
      logs.push(message)
    },
    ...overrides
  }

  return { deps, app, dialog, access, logs, userDataDir }
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop()

    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  }
})

describe('resolveMacAppBundle', () => {
  it('resolves packaged .app bundles on macOS including App Translocation paths', () => {
    expect(resolveMacAppBundle('/Applications/Hermes Agent.app/Contents/MacOS/Hermes Agent', 'darwin')).toBe(
      '/Applications/Hermes Agent.app'
    )
    expect(
      resolveMacAppBundle(
        '/private/var/folders/ab/cd/T/AppTranslocation/1F2E3D4C/d/Hermes Agent.app/Contents/MacOS/Hermes Agent',
        'darwin'
      )
    ).toBe('/private/var/folders/ab/cd/T/AppTranslocation/1F2E3D4C/d/Hermes Agent.app')
  })

  it('returns null off macOS or for non-bundle executables', () => {
    expect(resolveMacAppBundle('/Applications/Hermes Agent.app/Contents/MacOS/Hermes Agent', 'linux')).toBeNull()
    expect(resolveMacAppBundle('/Applications/Hermes Agent.app/Contents/MacOS/Hermes Agent', 'win32')).toBeNull()
    expect(resolveMacAppBundle('/usr/local/bin/electron', 'darwin')).toBeNull()
    expect(resolveMacAppBundle('', 'darwin')).toBeNull()
  })
})

describe('classifyMacBundleLocation', () => {
  const writable = () => undefined

  it('detects App Translocation and read-only volumes while ignoring EACCES', () => {
    expect(
      classifyMacBundleLocation('/private/var/folders/x/T/AppTranslocation/9A8B/d/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: writable
      })
    ).toBe('app-translocation')

    const erofs = () => {
      throw Object.assign(new Error('read-only file system'), { code: 'EROFS' })
    }

    expect(
      classifyMacBundleLocation('/Volumes/Hermes Agent/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: erofs
      })
    ).toBe('read-only-volume')

    const erofsBundleOnly = (target: string) => {
      if (target.endsWith('.app')) {
        throw Object.assign(new Error('read-only file system'), { code: 'EROFS' })
      }
    }

    expect(
      classifyMacBundleLocation('/Volumes/Hermes Agent/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: erofsBundleOnly
      })
    ).toBe('read-only-volume')

    const eacces = () => {
      throw Object.assign(new Error('permission denied'), { code: 'EACCES' })
    }

    expect(
      classifyMacBundleLocation('/opt/hermes/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: eacces
      })
    ).toBeNull()
  })

  it('detects Downloads and Desktop folders without matching sibling prefixes or CI runner temp dirs', () => {
    expect(
      classifyMacBundleLocation('/Users/alice/Downloads/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: writable
      })
    ).toBe('downloads-folder')
    expect(
      classifyMacBundleLocation('/Users/alice/Downloads/unzipped/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: writable
      })
    ).toBe('downloads-folder')
    expect(
      classifyMacBundleLocation('/Users/alice/Desktop/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: writable
      })
    ).toBe('desktop-folder')
    expect(
      classifyMacBundleLocation('/Users/alice/Downloads-keep/Hermes Agent.app', {
        homeDir: '/Users/alice',
        access: writable
      })
    ).toBeNull()
    expect(
      classifyMacBundleLocation('/Users/runner/work/_temp/hermes-bundled-e2e/apps/Hermes.app', {
        homeDir: '/Users/runner',
        access: writable
      })
    ).toBeNull()
  })
})

describe('guardMacApplicationsLaunch', () => {
  it('skips non-macOS, dev runs, source checkout builds, and apps already in Applications', async () => {
    for (const overrides of [
      { platform: 'linux' as const },
      { platform: 'win32' as const },
      { isPackaged: false },
      { installStamp: null },
      { installStamp: { payload: 'bootstrap' as const, updateMechanism: 'self' as const } }
    ]) {
      const { deps, app, dialog } = makeDeps(overrides)
      const result = await guardMacApplicationsLaunch(deps)

      expect(result.action).toBe('continue')
      expect(app.isInApplicationsFolder).not.toHaveBeenCalled()
      expect(dialog.showMessageBox).not.toHaveBeenCalled()
      expect(app.moveToApplicationsFolder).not.toHaveBeenCalled()
    }

    const inApps = makeDeps()
    inApps.app.getPath = vi.fn((name: string) =>
      name === 'exe' ? '/Applications/Hermes Agent.app/Contents/MacOS/Hermes Agent' : inApps.userDataDir
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']
    inApps.app.isInApplicationsFolder.mockReturnValue(true)

    await expect(guardMacApplicationsLaunch(inApps.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'already-in-applications'
    })
    expect(inApps.dialog.showMessageBox).not.toHaveBeenCalled()
  })

  it.each([
    ['CI=true', { env: { CI: 'true' } }],
    ['GITHUB_ACTIONS=true', { env: { GITHUB_ACTIONS: 'true' } }],
    ['HERMES_DESKTOP_SKIP_QUIT_CONFIRM=1', { env: { HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1' } }],
    ['HERMES_DESKTOP_USER_DATA_DIR', { env: { HERMES_DESKTOP_USER_DATA_DIR: '/tmp/smoke-ud' } }],
    [
      'Playwright --remote-debugging-port',
      { argv: ['/Volumes/Hermes/Hermes.app/Contents/MacOS/Hermes', '--remote-debugging-port=0'] }
    ]
  ])('never prompts in automated or smoke environments (%s)', async (_label, overrides) => {
    const { deps, dialog, app } = makeDeps(overrides)
    const result = await guardMacApplicationsLaunch(deps)

    expect(result).toEqual({ action: 'continue', reason: 'automated-environment' })
    expect(dialog.showMessageBox).not.toHaveBeenCalled()
    expect(app.moveToApplicationsFolder).not.toHaveBeenCalled()
  })

  it('never prompts when ShipIt relaunches the macOS E2E upgrade app in a writable temp dir with empty env', async () => {
    const { deps, app, dialog } = makeDeps({
      env: {},
      argv: ['/Users/runner/work/_temp/hermes-bundled-e2e/apps/Hermes.app/Contents/MacOS/Hermes Agent'],
      homeDir: '/Users/runner',
      access: () => undefined
    })

    app.getPath = vi.fn((name: string) =>
      name === 'exe'
        ? '/Users/runner/work/_temp/hermes-bundled-e2e/apps/Hermes.app/Contents/MacOS/Hermes Agent'
        : makeTempUserData()
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'custom-writable-location'
    })
    expect(dialog.showMessageBox).not.toHaveBeenCalled()
    expect(app.moveToApplicationsFolder).not.toHaveBeenCalled()
  })

  it.each([
    [
      'mounted DMG',
      '/Volumes/Hermes Agent/Hermes Agent.app/Contents/MacOS/Hermes Agent',
      () => {
        throw Object.assign(new Error('read-only file system'), { code: 'EROFS' })
      },
      'read-only-volume'
    ],
    [
      'App Translocation',
      '/private/var/folders/aa/bb/T/AppTranslocation/1111-2222/d/Hermes Agent.app/Contents/MacOS/Hermes Agent',
      () => undefined,
      'app-translocation'
    ],
    [
      'Downloads folder',
      '/Users/alice/Downloads/Hermes Agent.app/Contents/MacOS/Hermes Agent',
      () => undefined,
      'downloads-folder'
    ],
    [
      'Desktop folder',
      '/Users/alice/Desktop/Hermes Agent.app/Contents/MacOS/Hermes Agent',
      () => undefined,
      'desktop-folder'
    ]
  ] as const)('offers and executes Move to Applications from %s', async (_label, exePath, access, expectedLocation) => {
    const { deps, app, dialog, userDataDir } = makeDeps({
      access,
      installStamp: { payload: 'light', updateMechanism: 'electron-updater' }
    })

    app.getPath = vi.fn((name: string) =>
      name === 'exe' ? exePath : userDataDir
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']

    let conflictHandler: ((type: 'exists' | 'existsAndRunning') => boolean) | undefined

    app.moveToApplicationsFolder.mockImplementation(options => {
      conflictHandler = options?.conflictHandler

      return true
    })

    const result = await guardMacApplicationsLaunch(deps)

    expect(result).toEqual({ action: 'relaunching', location: expectedLocation })
    expect(dialog.showMessageBox).toHaveBeenCalledOnce()
    expect(dialog.showMessageBox).toHaveBeenCalledWith(
      expect.objectContaining({
        buttons: ['Move to Applications', 'Keep running here'],
        defaultId: 0,
        cancelId: 1,
        checkboxLabel: "Don't ask again",
        checkboxChecked: false
      })
    )
    expect(conflictHandler?.('exists')).toBe(true)
    expect(conflictHandler?.('existsAndRunning')).toBe(false)
  })

  it('still prompts for App Translocation even if isInApplicationsFolder reports true', async () => {
    const { deps, app, dialog, userDataDir } = makeDeps({ access: () => undefined })

    app.isInApplicationsFolder.mockReturnValue(true)
    app.getPath = vi.fn((name: string) =>
      name === 'exe'
        ? '/private/var/folders/aa/bb/T/AppTranslocation/DEAD-BEEF/d/Hermes Agent.app/Contents/MacOS/Hermes Agent'
        : userDataDir
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'relaunching',
      location: 'app-translocation'
    })
    expect(dialog.showMessageBox).toHaveBeenCalledOnce()
  })

  it('keeps running without persisting when user declines without checking Do not ask again', async () => {
    const { deps, app, dialog, userDataDir } = makeDeps()

    dialog.showMessageBox.mockResolvedValue({ response: 1, checkboxChecked: false })

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'user-declined',
      location: 'read-only-volume',
      remembered: false
    })
    expect(app.moveToApplicationsFolder).not.toHaveBeenCalled()
    expect(fs.existsSync(macApplicationsGuardPreferencePath(userDataDir))).toBe(false)

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'user-declined',
      location: 'read-only-volume',
      remembered: false
    })
    expect(dialog.showMessageBox).toHaveBeenCalledTimes(2)
  })

  it('remembers dismissal across launches including fresh App Translocation UUIDs', async () => {
    const { deps, app, dialog, userDataDir } = makeDeps({ access: () => undefined })

    let exe = '/private/var/folders/aa/bb/T/AppTranslocation/UUID-FIRST/d/Hermes Agent.app/Contents/MacOS/Hermes Agent'

    app.getPath = vi.fn((name: string) =>
      name === 'exe' ? exe : userDataDir
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']
    dialog.showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: true })

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'user-declined',
      location: 'app-translocation',
      remembered: true
    })

    const saved = readMacApplicationsGuardPreference(userDataDir)

    expect(saved).toMatchObject({
      schemaVersion: 1,
      dismissed: true,
      reason: 'app-translocation',
      appVersion: '0.24.0'
    })

    exe = '/private/var/folders/aa/bb/T/AppTranslocation/UUID-SECOND/d/Hermes Agent.app/Contents/MacOS/Hermes Agent'

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'dismissed',
      location: 'app-translocation'
    })
    expect(dialog.showMessageBox).toHaveBeenCalledTimes(1)
  })

  it('handles existsAndRunning conflict, auth cancellation, and thrown move errors without trapping startup', async () => {
    const conflictCase = makeDeps()

    conflictCase.app.moveToApplicationsFolder.mockImplementation(options => {
      options?.conflictHandler?.('existsAndRunning')

      return false
    })

    await expect(guardMacApplicationsLaunch(conflictCase.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'conflict-running',
      location: 'read-only-volume'
    })
    expect(conflictCase.dialog.showMessageBox).toHaveBeenCalledTimes(2)
    expect(conflictCase.dialog.showMessageBox).toHaveBeenLastCalledWith(
      expect.objectContaining({
        message: 'Another copy of Hermes Agent is already running in Applications',
        buttons: ['Continue']
      })
    )
    expect(readMacApplicationsGuardPreference(conflictCase.userDataDir)).toBeNull()

    const cancelCase = makeDeps()

    cancelCase.app.moveToApplicationsFolder.mockReturnValue(false)

    await expect(guardMacApplicationsLaunch(cancelCase.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'move-cancelled',
      location: 'read-only-volume'
    })
    expect(cancelCase.dialog.showMessageBox).toHaveBeenCalledTimes(1)

    const errorCase = makeDeps()

    errorCase.app.moveToApplicationsFolder.mockImplementation(() => {
      throw new Error('ENOSPC: no space left on device')
    })

    await expect(guardMacApplicationsLaunch(errorCase.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'move-failed',
      location: 'read-only-volume',
      error: 'ENOSPC: no space left on device'
    })
    expect(errorCase.dialog.showMessageBox).toHaveBeenCalledTimes(2)
    expect(errorCase.dialog.showMessageBox).toHaveBeenLastCalledWith(
      expect.objectContaining({
        message: 'Hermes Agent could not move itself to Applications',
        detail: expect.stringContaining('ENOSPC: no space left on device')
      })
    )
    expect(readMacApplicationsGuardPreference(errorCase.userDataDir)).toBeNull()
  })

  it('recovers cleanly from corrupt saved preferences and unexpected startup errors', async () => {
    const { deps, dialog, userDataDir } = makeDeps()

    fs.writeFileSync(path.join(userDataDir, MAC_APPLICATIONS_GUARD_FILENAME), '{corrupt', 'utf8')
    dialog.showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: false })

    await expect(guardMacApplicationsLaunch(deps)).resolves.toMatchObject({
      action: 'continue',
      reason: 'user-declined'
    })

    const broken = makeDeps()

    broken.app.isInApplicationsFolder.mockImplementation(() => {
      throw new Error('NSBundle unavailable')
    })

    await expect(guardMacApplicationsLaunch(broken.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'error',
      error: 'NSBundle unavailable'
    })
    expect(broken.logs.some(line => line.includes('NSBundle unavailable'))).toBe(true)
  })
})
