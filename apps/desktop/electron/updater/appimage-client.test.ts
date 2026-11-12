import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

const { client } = vi.hoisted(() => ({
  client: {
    autoDownload: true,
    autoInstallOnAppQuit: true,
    autoRunAppAfterInstall: false,
    channel: '',
    allowPrerelease: true,
    allowDowngrade: true,
    on: vi.fn(),
    setFeedURL: vi.fn(),
    checkForUpdates: vi.fn(async () => null)
  }
}))

vi.mock('electron-updater', () => ({
  default: {
    AppImageUpdater: class {
      constructor() {
        return client
      }
    }
  }
}))

import { createAppImageStrategy, type RelaunchHost, relaunchUpdatedAppImage } from './appimage-client'

afterEach((): void => {
  vi.clearAllMocks()
})

const IMAGE = '/home/u/Applications/Hermes-0.99.0-linux-arm64.AppImage'

function deps(feedBaseUrl = '', channel: 'stable' | 'canary' = 'stable'): Parameters<typeof createAppImageStrategy>[0] {
  return {
    channel,
    light: false,
    feedBaseUrl,
    appVersion: '0.99.0',
    log: vi.fn(),
    emitProgress: vi.fn(),
    beforeInstall: vi.fn(),
    onInstallFailure: vi.fn()
  }
}

describe('AppImage client wiring', () => {
  it('uses the baked app-update.yml by default and forbids implicit installs or downgrades', async () => {
    const strategy = createAppImageStrategy(deps(), IMAGE, 'arm64')
    expect(client.setFeedURL).not.toHaveBeenCalled()
    await expect(strategy.check()).rejects.toThrow('not active')
    expect(client.autoDownload).toBe(false)
    expect(client.autoInstallOnAppQuit).toBe(false)
    expect(client.autoRunAppAfterInstall).toBe(true)
    expect(client.allowDowngrade).toBe(false)
    expect(client.channel).toBe('stable')
    expect(client.allowPrerelease).toBe(false)
  })

  it.each(['stable', 'canary'] as const)('overrides the %s feed with the directory the release publishes', channel => {
    createAppImageStrategy(deps('https://updates.example/', channel), IMAGE, 'arm64')
    expect(client.setFeedURL).toHaveBeenCalledWith({
      provider: 'generic',
      url: `https://updates.example/releases/linux/${channel}/`,
      channel
    })
    expect(client.channel).toBe(channel)
    expect(client.allowPrerelease).toBe(channel === 'canary')
  })

  it('takes a loopback test feed but never a plain-http remote one', () => {
    createAppImageStrategy(deps('http://127.0.0.1:8765', 'canary'), IMAGE, 'arm64')
    expect(client.setFeedURL).toHaveBeenCalledWith({
      provider: 'generic',
      url: 'http://127.0.0.1:8765/releases/linux/canary/',
      channel: 'canary'
    })
    expect(() => createAppImageStrategy(deps('http://updates.example'), IMAGE, 'arm64')).toThrow('HTTPS')
  })
})

describe('relaunchUpdatedAppImage', () => {
  const mount = '/tmp/.mount_HermesAb12'

  function launch(env: NodeJS.ProcessEnv) {
    const dir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'appimage-relaunch-'))
    const target: string = path.join(dir, 'Hermes-0.99.1-linux-arm64.AppImage')
    fs.writeFileSync(target, '#!/bin/sh\n', { mode: 0o755 })
    const spawn = vi.fn() as unknown as RelaunchHost['spawn']

    relaunchUpdatedAppImage(target, env, {
      argv: [`${mount}/hermes`, '--no-sandbox', '--remote-debugging-port=9222', 'hermes://open'],
      execPath: `${mount}/hermes`,
      spawn
    })

    return { target, spawn: vi.mocked(spawn) }
  }

  it('hands the new image to the after-exit relauncher with launch flags and without this mount', () => {
    const { target, spawn } = launch({
      APPIMAGE: IMAGE,
      APPDIR: mount,
      PATH: `${mount}/usr/bin:/usr/bin`,
      LD_LIBRARY_PATH: `${mount}/usr/lib`,
      HERMES_DESKTOP_READY_FD: '5',
      APPIMAGE_SILENT_INSTALL: 'true'
    })

    expect(spawn).toHaveBeenCalledTimes(1)
    const [file, args, env] = spawn.mock.calls[0]
    expect(file).toBe(target)
    expect(args).toEqual(['--no-sandbox', 'hermes://open'])
    expect(env).toMatchObject({ PATH: '/usr/bin', APPIMAGE_SILENT_INSTALL: 'true' })
    expect(env).not.toHaveProperty('APPDIR')
    expect(env).not.toHaveProperty('LD_LIBRARY_PATH')
    expect(env).not.toHaveProperty('HERMES_DESKTOP_READY_FD')
    expect(env).not.toHaveProperty('APPIMAGE_EXTRACT_AND_RUN')
  })

  it('starts the new image without FUSE when this one ran extracted', () => {
    const extracted = '/tmp/appimage_extracted_0123abcd'
    const { spawn } = launch({ APPIMAGE: IMAGE, APPDIR: extracted, PATH: `${extracted}/usr/bin:/usr/bin` })
    expect(spawn.mock.calls[0][2]).toMatchObject({ APPIMAGE_EXTRACT_AND_RUN: '1' })
  })

  it('refuses before quitting when the new image cannot be started', () => {
    const spawn = vi.fn() as unknown as RelaunchHost['spawn']
    expect(() =>
      relaunchUpdatedAppImage(
        '/nonexistent/Hermes.AppImage',
        { APPIMAGE: IMAGE, APPDIR: mount },
        {
          argv: [],
          execPath: `${mount}/hermes`,
          spawn
        }
      )
    ).toThrow('cannot be started')
    expect(spawn).not.toHaveBeenCalled()
  })
})
