import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { appImageReplaceable, AppImageStrategy, type AppImageStrategyDeps } from './appimage'

function fixture(overrides: Partial<AppImageStrategyDeps> = {}) {
  const events: string[] = []
  const emitter = new EventEmitter()

  const info = {
    version: '0.99.1',
    files: [{ url: 'Hermes-0.99.1-linux-arm64.AppImage', sha512: 'x' }],
    releaseDate: '',
    path: '',
    sha512: ''
  }

  const deps: AppImageStrategyDeps = {
    updater: {
      checkForUpdates: vi.fn(async () => {
        events.push('check')

        return { isUpdateAvailable: true, updateInfo: info, versionInfo: info }
      }),
      downloadUpdate: vi.fn(async () => {
        events.push('download')

        return []
      }),
      quitAndInstall: vi.fn(() => {
        events.push('install')
      }),
      on: emitter.on.bind(emitter) as AppImageStrategyDeps['updater']['on'],
      removeListener: emitter.removeListener.bind(emitter) as AppImageStrategyDeps['updater']['removeListener']
    },
    channel: 'canary',
    appVersion: '0.99.0',
    appImage: '/home/u/Applications/Hermes-0.99.0-linux-arm64.AppImage',
    feedUrl: 'https://updates.example/releases/linux/canary/',
    canReplace: () => true,
    beforeInstall: vi.fn(async () => {
      events.push('stop')
    }),
    onInstallFailure: vi.fn(async () => {
      events.push('restore')
    }),
    emitProgress: vi.fn(),
    ...overrides
  }

  return { deps, events, emitter, strategy: new AppImageStrategy(deps) }
}

describe('AppImage strategy', () => {
  it('checks without downloading, then downloads before stopping the backends and installs once', async () => {
    const { strategy, events, emitter } = fixture()
    expect(await strategy.check()).toMatchObject({
      supported: true,
      mechanism: 'electron-updater',
      channel: 'canary',
      currentVersion: '0.99.0',
      latestTag: 'v0.99.1',
      updateAvailable: true
    })
    expect(events).toEqual(['check'])
    events.length = 0
    expect(await strategy.apply()).toMatchObject({ ok: true, handedOff: true, bundled: true })
    expect(events).toEqual(['check', 'download', 'stop', 'install'])
    expect(emitter.listenerCount('download-progress')).toBe(0)
    expect(emitter.listenerCount('error')).toBe(0)
  })

  it('offers the new AppImage from the feed instead of downloading when its folder is not writable', async () => {
    const { strategy, events, deps } = fixture({ canReplace: () => false })
    const result = await strategy.apply()
    expect(result).toMatchObject({
      ok: false,
      error: 'appimage-not-writable',
      downloadUrl: 'https://updates.example/releases/linux/canary/Hermes-0.99.1-linux-arm64.AppImage'
    })
    expect(result.message).toContain('/home/u/Applications is not writable')
    expect(result.message).toContain(String(result.downloadUrl))
    expect(events).toEqual(['check'])
    expect(deps.onInstallFailure).not.toHaveBeenCalled()
  })

  it('restarts the backends when the replacement fails, which electron-updater reports only as an event', async () => {
    const { strategy, events, deps, emitter } = fixture()
    vi.mocked(deps.updater.quitAndInstall).mockImplementationOnce(() => {
      events.push('install')
      emitter.emit('error', new Error('EACCES: permission denied, unlink'))
    })
    await expect(strategy.apply()).rejects.toThrow('EACCES')
    expect(events).toEqual(['check', 'download', 'stop', 'install', 'restore'])
    expect(emitter.listenerCount('error')).toBe(0)
  })

  it('keeps the backends running when the download fails', async () => {
    const { strategy, events, deps } = fixture()
    vi.mocked(deps.updater.downloadUpdate).mockRejectedValueOnce(new Error('sha512 checksum mismatch'))
    await expect(strategy.apply()).rejects.toThrow('sha512')
    expect(events).toEqual(['check'])
  })

  it('does nothing when the feed has no newer release', async () => {
    const { deps, strategy, events } = fixture()
    const info = { version: '0.99.0', files: [], releaseDate: '', path: '', sha512: '' }
    vi.mocked(deps.updater.checkForUpdates).mockResolvedValue({
      isUpdateAvailable: false,
      updateInfo: info,
      versionInfo: info
    })
    expect(await strategy.apply()).toEqual({ ok: true, mechanism: 'electron-updater' })
    expect(events).toEqual([])
  })

  it('names an inactive updater instead of reporting no update', async () => {
    const { deps, strategy } = fixture()
    vi.mocked(deps.updater.checkForUpdates).mockResolvedValue(null)
    await expect(strategy.check()).rejects.toThrow('AppImage updater is not active')
  })
})

describe('appImageReplaceable', () => {
  it('needs write access to the folder holding the AppImage, not to the file', () => {
    const dir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'appimage-dir-'))

    try {
      const image: string = path.join(dir, 'Hermes.AppImage')
      fs.writeFileSync(image, '')
      fs.chmodSync(image, 0o555)
      expect(appImageReplaceable(image)).toBe(true)
      fs.chmodSync(dir, 0o555)
      expect(appImageReplaceable(image)).toBe(process.getuid?.() === 0)
    } finally {
      fs.chmodSync(dir, 0o755)
      fs.rmSync(dir, { recursive: true, force: true })
    }
  })
})
