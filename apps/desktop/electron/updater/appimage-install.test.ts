import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { AppAdapter } from 'electron-updater/out/AppAdapter'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { RelaunchingAppImageUpdater } from './appimage-client'

let dir: string
let appImage: string | undefined

beforeEach((): void => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'appimage-install-'))
  appImage = process.env.APPIMAGE
})

afterEach((): void => {
  process.env.APPIMAGE = appImage

  if (appImage === undefined) {
    delete process.env.APPIMAGE
  }

  fs.rmSync(dir, { recursive: true, force: true })
})

function app(): AppAdapter {
  return {
    version: '0.99.0',
    name: 'Hermes',
    isPackaged: true,
    appUpdateConfigPath: path.join(dir, 'app-update.yml'),
    userDataPath: dir,
    baseCachePath: dir,
    whenReady: async () => undefined,
    relaunch: vi.fn(),
    quit: vi.fn(),
    onQuit: vi.fn()
  }
}

/** Exercise the real updater install entry point with a local download fixture. */
function install(oldName: string, newName: string, failure?: 'missing-download' | 'blocked-destination') {
  const old: string = path.join(dir, 'Applications', oldName)
  const downloaded: string = path.join(dir, 'pending', newName)
  fs.mkdirSync(path.dirname(old))
  fs.mkdirSync(path.dirname(downloaded))
  fs.writeFileSync(old, 'old build', { mode: 0o755 })
  fs.writeFileSync(downloaded, 'new build', { mode: 0o755 })
  process.env.APPIMAGE = old

  const relaunch = vi.fn()
  const updater = new RelaunchingAppImageUpdater(relaunch, app())
  const errors: Error[] = []
  updater.on('error', (error: Error) => errors.push(error))
  Object.assign(updater, { downloadedUpdateHelper: { file: downloaded, downloadedFileInfo: {} } })

  if (failure === 'missing-download') {
    fs.unlinkSync(downloaded)
  } else if (failure === 'blocked-destination') {
    // A directory at the versioned destination blocks rename. Give GNU mv no
    // way to move into that directory either: its nested target is a directory.
    fs.mkdirSync(path.join(path.dirname(old), newName, newName), { recursive: true })
  }

  const installed: boolean = updater.install(false, true)

  return { old, downloaded, installed, relaunch, errors }
}

it('replaces an unversioned AppImage in place and relaunches it through Hermes, not a direct spawn', () => {
  const { old, installed, relaunch, errors } = install('Hermes.AppImage', 'Hermes-0.99.1-linux-arm64.AppImage')
  expect(errors).toEqual([])
  expect(installed).toBe(true)
  expect(fs.readFileSync(old, 'utf8')).toBe('new build')
  expect(relaunch).toHaveBeenCalledWith(old, expect.objectContaining({ APPIMAGE_SILENT_INSTALL: 'true' }))
})

it('relaunches the new file when a versioned AppImage takes the new name', () => {
  const { old, installed, relaunch } = install(
    'Hermes-0.99.0-linux-arm64.AppImage',
    'Hermes-0.99.1-linux-arm64.AppImage'
  )

  const renamed: string = path.join(path.dirname(old), 'Hermes-0.99.1-linux-arm64.AppImage')
  expect(installed).toBe(true)
  expect(fs.existsSync(old)).toBe(false)
  expect(fs.readFileSync(renamed, 'utf8')).toBe('new build')
  expect(relaunch).toHaveBeenCalledWith(renamed, expect.any(Object))
})

it('keeps the previous AppImage when the cached download disappears before install', () => {
  const { old, installed, relaunch, errors } = install(
    'Hermes.AppImage',
    'Hermes-0.99.1-linux-arm64.AppImage',
    'missing-download'
  )

  expect(installed).toBe(false)
  expect(errors).toHaveLength(1)
  expect(relaunch).not.toHaveBeenCalled()
  expect(fs.readFileSync(old, 'utf8')).toBe('old build')
  expect(fs.statSync(old).mode & 0o111).toBe(0o111)
})

it('keeps the previous AppImage when replacing the destination fails with the download present', () => {
  const oldName = 'Hermes-0.99.0-linux-arm64.AppImage'
  const newName = 'Hermes-0.99.1-linux-arm64.AppImage'
  const { old, downloaded, installed, relaunch, errors } = install(oldName, newName, 'blocked-destination')

  expect(installed).toBe(false)
  expect(errors).toHaveLength(1)
  expect(relaunch).not.toHaveBeenCalled()
  expect(fs.readFileSync(downloaded, 'utf8')).toBe('new build')
  expect(fs.readFileSync(old, 'utf8')).toBe('old build')
  expect(fs.statSync(old).mode & 0o111).toBe(0o111)
  expect(fs.readdirSync(path.dirname(old)).sort()).toEqual([oldName, newName])
})
