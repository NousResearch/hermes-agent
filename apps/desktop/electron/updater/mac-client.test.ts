import { compare } from 'semver'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { client, defaultUpdateSupport } = vi.hoisted(() => {
  const defaultUpdateSupport = vi.fn(async (_updateInfo: { version: string }): Promise<boolean> => true)

  return {
    defaultUpdateSupport,
    client: {
      autoDownload: true,
      autoInstallOnAppQuit: true,
      autoRunAppAfterInstall: false,
      channel: '',
      allowPrerelease: true,
      allowDowngrade: true,
      currentVersion: { version: '9.9.9' },
      isUpdateSupported: defaultUpdateSupport,
      on: vi.fn(),
      setFeedURL: vi.fn(),
      checkForUpdates: vi.fn(async () => null)
    }
  }
})

vi.mock('electron', () => ({ autoUpdater: {} }))
vi.mock('electron-updater', () => ({
  default: {
    MacUpdater: class {
      constructor() {
        return client
      }
    }
  }
}))

import { createMacStrategy } from './mac-client'

afterEach((): void => {
  vi.clearAllMocks()
  client.currentVersion = { version: '9.9.9' }
  client.isUpdateSupported = defaultUpdateSupport
})

function deps(
  feedBaseUrl = '',
  light = false,
  channel: 'stable' | 'canary' = 'stable'
): Parameters<typeof createMacStrategy>[0] {
  return {
    channel,
    light,
    feedBaseUrl,
    appVersion: '0.28.0',
    log: vi.fn(),
    emitProgress: vi.fn(),
    beforeInstall: vi.fn(),
    onInstallFailure: vi.fn()
  }
}

describe('macOS client wiring', () => {
  it('uses the generated provider by default and forbids implicit installs or downgrades', async () => {
    const strategy = createMacStrategy(deps())
    expect(client.setFeedURL).not.toHaveBeenCalled()
    await expect(strategy.check()).rejects.toThrow('not active')
    expect(client.autoDownload).toBe(false)
    expect(client.autoInstallOnAppQuit).toBe(false)
    expect(client.autoRunAppAfterInstall).toBe(true)
    expect(client.allowDowngrade).toBe(false)
    expect(client.channel).toBe('stable')
    expect(client.allowPrerelease).toBe(false)
  })

  it('pins a neutral immutable descriptor instead of interpolating the dynamic name', (): void => {
    createMacStrategy({
      ...deps('https://updates.example'),
      channel: 'unknown-preview',
      feed: { url: 'https://updates.example/releases/channel-builds/abc/darwin/latest-mac.yml', channel: 'latest' }
    })
    expect(client.setFeedURL).toHaveBeenCalledWith({
      provider: 'generic',
      url: 'https://updates.example/releases/channel-builds/abc/darwin/',
      channel: 'latest'
    })
    expect(client.allowDowngrade).toBe(false)
    expect(client.currentVersion.version).toBe('0.0.0')
    expect(client.autoDownload).toBe(false)
    expect((): void => {
      createMacStrategy({
        ...deps('https://updates.example'),
        feed: { url: 'https://other.example/latest-mac.yml', channel: 'latest' }
      })
    }).toThrow('authority')
  })

  it('uses the installed version as the floor for signed stale, equal, and newer channel targets', async (): Promise<void> => {
    const installedVersion = '0.0.2'

    const controls = [
      { label: 'stale', version: '0.0.1', updateAvailable: false },
      { label: 'equal reinstall', version: installedVersion, updateAvailable: true },
      { label: 'newer', version: '0.0.3', updateAvailable: true }
    ] as const

    for (const control of controls) {
      client.currentVersion = { version: installedVersion }
      client.isUpdateSupported = defaultUpdateSupport
      client.checkForUpdates.mockImplementationOnce(async () => {
        // This models electron-updater after the immutable metadata and signed
        // bundle have passed channel admission. The synthetic comparison base
        // must not erase the installed-version floor.
        const info = {
          version: control.version,
          files: [],
          releaseDate: '',
          path: '',
          sha512: 'signed-fixture'
        }

        const supported = await client.isUpdateSupported(info)
        const updateAvailable = supported && compare(control.version, client.currentVersion.version) > 0

        return { isUpdateAvailable: updateAvailable, updateInfo: info, versionInfo: info }
      })

      const strategy = createMacStrategy({
        ...deps('https://updates.example'),
        appVersion: installedVersion,
        expectedVersion: control.version,
        feed: {
          url: 'https://updates.example/releases/channel-builds/abc/darwin/latest-mac.yml',
          channel: 'latest'
        }
      })

      expect((await strategy.check()).updateAvailable, control.label).toBe(control.updateAvailable)
    }

    expect(defaultUpdateSupport).toHaveBeenCalledTimes(2)
  })

  it('checks a newer channel head when SemVer build metadata has equal precedence', async (): Promise<void> => {
    const version = '0.21.4+canary.20260922T001500Z'
    client.checkForUpdates.mockImplementationOnce(async () => {
      expect(client.currentVersion.version).toBe('0.0.0')
      const info = { version, files: [], releaseDate: '', path: '', sha512: '' }

      return { isUpdateAvailable: true, updateInfo: info, versionInfo: info }
    })

    const strategy = createMacStrategy({
      ...deps('https://updates.example', false, 'canary'),
      appVersion: '0.21.4+canary.20260922T001400Z',
      expectedVersion: version,
      feed: {
        url: 'https://updates.example/releases/channel-builds/abc/darwin/latest-mac.yml',
        channel: 'latest'
      }
    })

    await expect(strategy.check()).resolves.toMatchObject({ updateAvailable: true, latestTag: `v${version}` })
  })

  it('overrides the provider with the same variant/channel path as the publisher', () => {
    createMacStrategy(deps('https://updates.example/', true, 'canary'))
    expect(client.setFeedURL).toHaveBeenCalledWith({
      provider: 'generic',
      url: 'https://updates.example/releases/darwin/light/canary/',
      channel: 'canary'
    })
    expect(client.allowPrerelease).toBe(true)
    expect(() => createMacStrategy(deps('http://untrusted.example'))).toThrow('HTTPS')
  })
})
