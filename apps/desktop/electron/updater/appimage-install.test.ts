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
  vi.restoreAllMocks()
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
function install(
  oldName: string,
  newName: string,
  fixture?: 'missing-download' | 'blocked-destination' | 'dotted-path'
) {
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

  if (fixture === 'missing-download') {
    fs.unlinkSync(downloaded)
  } else if (fixture === 'blocked-destination') {
    // A directory at the versioned destination blocks rename. Give GNU mv no
    // way to move into that directory either: its nested target is a directory.
    fs.mkdirSync(path.join(path.dirname(old), newName, newName), { recursive: true })
  } else if (fixture === 'dotted-path') {
    process.env.APPIMAGE = `${path.dirname(old)}/./${oldName}`
  }

  const installed: boolean = updater.install(false, true)

  return { old, downloaded, installed, relaunch, errors }
}

/** The download cache on another filesystem: renaming out of it fails as rename(2) does. */
function crossDevice(): void {
  const rename = fs.renameSync

  vi.spyOn(fs, 'renameSync').mockImplementation((from: fs.PathLike, to: fs.PathLike): void => {
    if (String(from).startsWith(path.join(dir, 'pending'))) {
      throw Object.assign(new Error(`EXDEV: cross-device link not permitted, rename '${from}'`), { code: 'EXDEV' })
    }

    rename(from, to)
  })
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

it('installs over an APPIMAGE path spelled with ./ that already has the download name', () => {
  const name = 'Hermes-0.99.1-linux-arm64.AppImage'
  const { old, installed, relaunch, errors } = install(name, name, 'dotted-path')

  expect(errors).toEqual([])
  expect(installed).toBe(true)
  expect(fs.readFileSync(old, 'utf8')).toBe('new build')
  expect(relaunch).toHaveBeenCalledWith(process.env.APPIMAGE, expect.any(Object))
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

it('copies a download from another filesystem beside the AppImage, then swaps it in', () => {
  crossDevice()

  const { old, downloaded, installed, relaunch, errors } = install(
    'Hermes.AppImage',
    'Hermes-0.99.1-linux-arm64.AppImage'
  )

  expect(errors).toEqual([])
  expect(installed).toBe(true)
  expect(fs.readFileSync(old, 'utf8')).toBe('new build')
  expect(fs.statSync(old).mode & 0o111).toBe(0o111)
  expect(fs.existsSync(downloaded)).toBe(false)
  expect(fs.readdirSync(path.dirname(old))).toEqual(['Hermes.AppImage'])
  expect(relaunch).toHaveBeenCalledWith(old, expect.any(Object))
})

it('keeps the AppImage whole when the disk fills while copying from another filesystem', () => {
  crossDevice()
  vi.spyOn(fs, 'copyFileSync').mockImplementation((_from: fs.PathLike, to: fs.PathLike): void => {
    fs.writeFileSync(to, 'half a build')
    throw Object.assign(new Error('ENOSPC: no space left on device, copyfile'), { code: 'ENOSPC' })
  })

  const { old, downloaded, installed, relaunch, errors } = install(
    'Hermes.AppImage',
    'Hermes-0.99.1-linux-arm64.AppImage'
  )

  expect(installed).toBe(false)
  expect(errors).toHaveLength(1)
  expect(relaunch).not.toHaveBeenCalled()
  expect(fs.readFileSync(old, 'utf8')).toBe('old build')
  expect(fs.readFileSync(downloaded, 'utf8')).toBe('new build')
  expect(fs.readdirSync(path.dirname(old))).toEqual(['Hermes.AppImage'])
})

it('keeps the AppImage and the download, and no staging copy, when a cross-filesystem swap fails', () => {
  crossDevice()
  const oldName = 'Hermes-0.99.0-linux-arm64.AppImage'
  const newName = 'Hermes-0.99.1-linux-arm64.AppImage'
  const { old, downloaded, installed, relaunch, errors } = install(oldName, newName, 'blocked-destination')

  expect(installed).toBe(false)
  expect(errors).toHaveLength(1)
  expect(relaunch).not.toHaveBeenCalled()
  expect(fs.readFileSync(downloaded, 'utf8')).toBe('new build')
  expect(fs.readFileSync(old, 'utf8')).toBe('old build')
  expect(fs.readdirSync(path.dirname(old)).sort()).toEqual([oldName, newName])
})
