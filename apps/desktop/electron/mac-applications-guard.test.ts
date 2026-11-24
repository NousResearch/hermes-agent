import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  classifyMacBundleLocation,
  compareBundleVersions,
  ephemeralMacBundleLocation,
  guardMacApplicationsLaunch,
  MAC_APPLICATIONS_GUARD_FILENAME,
  type MacApplicationsGuardDeps,
  macApplicationsGuardPreferencePath,
  parseBundleShortVersionFromPlist,
  readBundleShortVersion,
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
    readBundleVersion: () => null,
    log: message => {
      logs.push(message)
    },
    ...overrides
  }

  return { deps, app, dialog, access, logs, userDataDir }
}

/** Installed bundles by path; a null version is a copy whose Info.plist gives none. */
function installedCopies(versions: Readonly<Record<string, string | null>>) {
  return {
    exists: (target: string) => target in versions,
    readBundleVersion: (target: string) => versions[target] ?? null
  }
}

/** Stands in for Electron: reports an 'exists' conflict to the guard and records its answer. */
function captureExistsDecision(app: ReturnType<typeof makeDeps>['app']): { value?: boolean } {
  const decision: { value?: boolean } = {}

  app.moveToApplicationsFolder.mockImplementation(options => {
    decision.value = options?.conflictHandler?.('exists')

    return Boolean(decision.value)
  })

  return decision
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

describe('ephemeralMacBundleLocation', () => {
  it('reports only what macOS can take away, not the prompt-only Downloads and Desktop rules', () => {
    for (const folder of ['Downloads', 'Desktop']) {
      const bundle = `/Users/alice/${folder}/Hermes Agent.app`

      expect(classifyMacBundleLocation(bundle, { homeDir: '/Users/alice', access: () => undefined })).not.toBeNull()
      expect(ephemeralMacBundleLocation(bundle, () => undefined)).toBeNull()
    }
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
    expect(conflictHandler?.('existsAndRunning')).toBe(false)
    // Light ships no CLI, and a translocated or read-only bundle never links one, so the prompt promises none.
    expect(dialog.showMessageBox).toHaveBeenCalledWith(
      expect.objectContaining({ detail: expect.not.stringContaining('command-line') })
    )
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

    expect(JSON.parse(fs.readFileSync(macApplicationsGuardPreferencePath(userDataDir), 'utf8'))).toEqual({
      dismissed: true
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

  it('refuses silent downgrade when /Applications or ~/Applications holds a newer version, and allows equal or newer versions', async () => {
    const downgrade = makeDeps(
      installedCopies({
        '/Volumes/Hermes Agent/Hermes Agent.app': '0.22.0',
        '/Applications/Hermes Agent.app': '0.24.0'
      })
    )

    const downgradeDecision = captureExistsDecision(downgrade.app)

    await expect(guardMacApplicationsLaunch(downgrade.deps)).resolves.toEqual({
      action: 'continue',
      reason: 'conflict-newer-installed',
      location: 'read-only-volume'
    })
    expect(downgradeDecision.value).toBe(false)
    expect(downgrade.dialog.showMessageBox).toHaveBeenCalledTimes(2)
    expect(downgrade.dialog.showMessageBox).toHaveBeenLastCalledWith(
      expect.objectContaining({
        message: 'A newer version of Hermes Agent is already in Applications',
        detail: expect.stringContaining('0.22.0')
      })
    )
    expect(downgrade.logs.some(line => line.includes('0.22.0') && line.includes('0.24.0'))).toBe(true)
    expect(readMacApplicationsGuardPreference(downgrade.userDataDir)).toBeNull()

    const canary = makeDeps(
      installedCopies({
        '/Volumes/Hermes Agent/Hermes Agent.app': '0.24.0+canary.20261120T090000Z',
        '/Applications/Hermes Agent.app': '0.24.0+canary.20261124T090000Z'
      })
    )

    const canaryDecision = captureExistsDecision(canary.app)

    await expect(guardMacApplicationsLaunch(canary.deps)).resolves.toMatchObject({
      action: 'continue',
      reason: 'conflict-newer-installed'
    })
    expect(canaryDecision.value).toBe(false)

    for (const installed of [
      { '/Users/alice/Applications/Hermes Agent.app': '0.24.0' },
      // Electron may replace the ~/Applications copy even when /Applications holds an older one.
      { '/Applications/Hermes Agent.app': '0.20.0', '/Users/alice/Applications/Hermes Agent.app': '0.30.0' }
    ]) {
      const userApps = makeDeps(
        installedCopies({ '/Volumes/Hermes Agent/Hermes Agent.app': '0.24.0-rc.1', ...installed })
      )

      const userAppsDecision = captureExistsDecision(userApps.app)

      await expect(guardMacApplicationsLaunch(userApps.deps)).resolves.toMatchObject({
        action: 'continue',
        reason: 'conflict-newer-installed'
      })
      expect(userAppsDecision.value).toBe(false)
    }

    for (const [runningVersion, existingVersion] of [
      ['0.24.0', '0.24.0'],
      ['0.25.0', '0.24.0']
    ] as const) {
      const translocated = '/private/var/folders/aa/bb/T/AppTranslocation/DITTO-COPY/d/Hermes Agent.app'

      const allowed = makeDeps({
        access: () => undefined,
        ...installedCopies({ [translocated]: runningVersion, '/Applications/Hermes Agent.app': existingVersion })
      })

      allowed.app.getPath = vi.fn((name: string) =>
        name === 'exe' ? `${translocated}/Contents/MacOS/Hermes Agent` : allowed.userDataDir
      ) as unknown as MacApplicationsGuardDeps['app']['getPath']

      const decision = captureExistsDecision(allowed.app)

      await expect(guardMacApplicationsLaunch(allowed.deps)).resolves.toEqual({
        action: 'relaunching',
        location: 'app-translocation'
      })
      expect(decision.value).toBe(true)
    }
  })

  it.each([
    ['an unreadable Info.plist', { '/Applications/Hermes Agent.app': null }],
    ['a version it cannot compare', { '/Applications/Hermes Agent.app': 'banana' }],
    ['a two-part version', { '/Applications/Hermes Agent.app': '0.23' }],
    [
      'an unreadable ~/Applications copy beside an older /Applications one',
      { '/Applications/Hermes Agent.app': '0.20.0', '/Users/alice/Applications/Hermes Agent.app': null }
    ],
    ['no copy at either Applications path', {}]
  ] as const)('refuses to replace an installed copy with %s', async (_label, installed) => {
    const { deps, app, dialog, logs, userDataDir } = makeDeps(
      installedCopies({ '/Volumes/Hermes Agent/Hermes Agent.app': '0.24.0', ...installed })
    )

    const decision = captureExistsDecision(app)

    await expect(guardMacApplicationsLaunch(deps)).resolves.toEqual({
      action: 'continue',
      reason: 'conflict-unverified-installed',
      location: 'read-only-volume'
    })
    expect(decision.value).toBe(false)
    expect(dialog.showMessageBox).toHaveBeenLastCalledWith(
      expect.objectContaining({
        message: 'Hermes Agent is already in Applications',
        detail: expect.stringContaining('could not be confirmed')
      })
    )
    expect(logs.some(line => line.includes('could not confirm'))).toBe(true)
    expect(readMacApplicationsGuardPreference(userDataDir)).toBeNull()
  })

  it('reports Do not ask again as remembered only when the preference was saved', async () => {
    const { deps, app, dialog, logs, userDataDir } = makeDeps()
    // userData below a regular file: the atomic write can't create its directory.
    const blocked = path.join(userDataDir, 'not-a-dir')

    fs.writeFileSync(blocked, '', 'utf8')
    app.getPath = vi.fn((name: string) =>
      name === 'exe' ? '/Volumes/Hermes Agent/Hermes Agent.app/Contents/MacOS/Hermes Agent' : path.join(blocked, 'ud')
    ) as unknown as MacApplicationsGuardDeps['app']['getPath']
    dialog.showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: true })

    await expect(guardMacApplicationsLaunch(deps)).resolves.toMatchObject({
      reason: 'user-declined',
      remembered: false
    })
    expect(logs.some(line => line.includes('could not save /Applications move preference'))).toBe(true)
    expect(logs.some(line => line.includes('remembered dismissal'))).toBe(false)
  })

  it.each([
    ['a bare dismissal', { dismissed: true }, false],
    ['an earlier build with extra fields', { schemaVersion: 1, dismissed: true, reason: 'read-only-volume' }, false],
    ['no dismissal', { dismissed: false }, true]
  ] as const)('reads the saved choice from the dismissal flag alone (%s)', async (_label, saved, prompts) => {
    const { deps, dialog, userDataDir } = makeDeps()

    fs.writeFileSync(macApplicationsGuardPreferencePath(userDataDir), JSON.stringify(saved), 'utf8')
    dialog.showMessageBox.mockResolvedValueOnce({ response: 1, checkboxChecked: false })

    await expect(guardMacApplicationsLaunch(deps)).resolves.toMatchObject({
      reason: prompts ? 'user-declined' : 'dismissed'
    })
    expect(dialog.showMessageBox).toHaveBeenCalledTimes(prompts ? 1 : 0)
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

describe('bundle version parsing and comparison', () => {
  it('parses CFBundleShortVersionString from XML Info.plist files', () => {
    const xml = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<plist version="1.0"><dict>',
      '  <key>CFBundleShortVersionString</key>',
      '  <string>0.24.0-canary.20261124</string>',
      '</dict></plist>'
    ].join('\n')

    expect(parseBundleShortVersionFromPlist(xml)).toBe('0.24.0-canary.20261124')
    expect(parseBundleShortVersionFromPlist('<plist><dict></dict></plist>')).toBeNull()

    const root = makeTempUserData()
    const bundleDir = path.join(root, 'Hermes Agent.app')

    fs.mkdirSync(path.join(bundleDir, 'Contents'), { recursive: true })
    fs.writeFileSync(path.join(bundleDir, 'Contents', 'Info.plist'), xml, 'utf8')

    expect(readBundleShortVersion(bundleDir)).toBe('0.24.0-canary.20261124')
    expect(readBundleShortVersion(path.join(root, 'Missing.app'))).toBeNull()
  })

  it('compares release, prerelease, numeric prerelease, and build-metadata versions', () => {
    expect(compareBundleVersions('0.22.0', '0.24.0')).toBe(-1)
    expect(compareBundleVersions('0.24.0', '0.24.0')).toBe(0)
    expect(compareBundleVersions('0.25.0', '0.24.0')).toBe(1)
    expect(compareBundleVersions('0.24.0-rc.1', '0.24.0')).toBe(-1)
    expect(compareBundleVersions('0.24.0', '0.24.0-rc.1')).toBe(1)
    expect(compareBundleVersions('0.24.0-rc.2', '0.24.0-rc.10')).toBe(-1)
    expect(compareBundleVersions('0.24.0-rc.10', '0.24.0-rc.2')).toBe(1)
    expect(compareBundleVersions('0.24.0+build.1', '0.24.0+build.1')).toBe(0)
    // Build metadata SemVer ignores still tells two builds of one core apart, so an unknown difference can't be ordered.
    expect(compareBundleVersions('0.24.0+build.1', '0.24.0+build.2')).toBeNull()
    expect(compareBundleVersions('0.24.0', '0.24.0+canary.20261124T001400Z')).toBeNull()
    expect(compareBundleVersions('0.24.0-canary.20261124T001400Z', '0.24.0-canary.20261124T001500Z')).toBe(-1)
    expect(compareBundleVersions('invalid', '0.24.0')).toBeNull()
  })

  it('orders canary and commit builds of one core by the UTC stamp in their build metadata', () => {
    expect(compareBundleVersions('0.24.0+canary.20261124T001400Z', '0.24.0+canary.20261125T001400Z')).toBe(-1)
    expect(compareBundleVersions('0.24.0+canary.20261125T001400Z', '0.24.0+canary.20261124T001400Z')).toBe(1)
    expect(compareBundleVersions('0.24.0+canary.20261124T001400Z', '0.24.0+canary.20261124T001400Z')).toBe(0)
    expect(compareBundleVersions('0.24.0+commit.20261124T001400Z', '0.24.0+commit.20261124T001500Z')).toBe(-1)
    expect(compareBundleVersions('0.25.0+canary.20261101T000000Z', '0.24.0+canary.20261124T001400Z')).toBe(1)
    expect(compareBundleVersions('0.24.0+canary.20261124T001400Z', '0.24.0+commit.20261125T001400Z')).toBeNull()
  })

  it.each(['0.23', '0.23.0-rc..1', '01.2.3', '0.24.0-', '0.24.0+', '0.24.0+build..1'])(
    'refuses to order %s, which is not a SemVer version',
    version => {
      expect(compareBundleVersions(version, '0.24.0')).toBeNull()
      expect(compareBundleVersions('0.24.0', version)).toBeNull()
    }
  )
})
