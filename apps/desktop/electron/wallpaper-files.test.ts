import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  fitWallpaperDimensions,
  isSupportedWallpaperPath,
  preferredWallpaperMaxEdge,
  readWallpaperFileAsset,
  readWallpaperSourceFile,
  removeWallpaperFile,
  WALLPAPER_PROTOCOL_PRIVILEGES,
  wallpaperAssetId,
  wallpaperAssetPredatesProfile,
  wallpaperFilePath,
  wallpaperFilePathFromAsset,
  writeWallpaperFile
} from './wallpaper-files'

test('wallpaper protocol can display images without exposing fetch or CORS access', () => {
  assert.deepEqual(WALLPAPER_PROTOCOL_PRIVILEGES, {
    secure: true,
    standard: true
  })
})

test('wallpaper assets are stable and profile-scoped', () => {
  assert.equal(wallpaperAssetId('default'), wallpaperAssetId('default'))
  assert.notEqual(wallpaperAssetId('default'), wallpaperAssetId('work'))
  assert.match(wallpaperFilePath('C:\\user-data', 'work'), /wallpapers[\\/]\w{24}\.jpg$/)
  assert.throws(() => wallpaperFilePath('C:\\user-data', '../work'))
  assert.throws(() => wallpaperFilePathFromAsset('C:\\user-data', '../secret'))
})

test('wallpaper protocol URLs carry an unguessable main-process access token when provided', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wallpaper-token-'))

  try {
    await writeWallpaperFile(root, 'default', Buffer.from('image'))

    const asset = await readWallpaperFileAsset(root, 'default', 'secret-token')

    assert.equal(new URL(asset?.url ?? '').searchParams.get('token'), 'secret-token')
    assert.equal(new URL(asset?.url ?? '').searchParams.get('v'), asset?.version)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('wallpaper input and import dimensions are bounded', () => {
  assert.equal(isSupportedWallpaperPath('photo.JPG'), true)
  assert.equal(isSupportedWallpaperPath('photo.webp'), true)
  assert.equal(isSupportedWallpaperPath('photo.svg'), false)
  assert.deepEqual(fitWallpaperDimensions(1920, 1080), { height: 1080, width: 1920 })
  assert.deepEqual(fitWallpaperDimensions(7680, 4320), { height: 2160, width: 3840 })
  assert.deepEqual(fitWallpaperDimensions(3000, 6000), { height: 3840, width: 1920 })
  assert.throws(() => fitWallpaperDimensions(0, 1080))
})

test('wallpaper import resolution follows the available displays within safe bounds', () => {
  assert.equal(preferredWallpaperMaxEdge([]), 1920)
  assert.equal(preferredWallpaperMaxEdge([{ height: 1440, scaleFactor: 1, width: 2560 }]), 2560)
  assert.equal(preferredWallpaperMaxEdge([{ height: 1080, scaleFactor: 2, width: 1920 }]), 3840)
  assert.equal(preferredWallpaperMaxEdge([{ height: 4320, scaleFactor: 1, width: 7680 }]), 3840)
  assert.equal(preferredWallpaperMaxEdge([{ height: 0, scaleFactor: 1, width: 0 }]), 1920)
  assert.throws(() => preferredWallpaperMaxEdge([], 3840, 1920))
})

test('wallpaper assets from an earlier profile lifetime are detected without timestamp false positives', () => {
  assert.equal(wallpaperAssetPredatesProfile(1_000, 5_000), true)
  assert.equal(wallpaperAssetPredatesProfile(4_500, 5_000), false)
  assert.equal(wallpaperAssetPredatesProfile(6_000, 5_000), false)
  assert.equal(wallpaperAssetPredatesProfile(1_000, 0), false)
})

test('wallpaper file replacement and removal clean up the profile asset', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wallpaper-'))

  try {
    const first = await writeWallpaperFile(root, 'default', Buffer.from('first'))
    const second = await writeWallpaperFile(root, 'default', Buffer.from('second-image'))

    assert.equal(first.filePath, second.filePath)
    assert.notEqual(first.url, second.url)
    assert.equal(fs.readFileSync(second.filePath, 'utf8'), 'second-image')
    assert.deepEqual(await readWallpaperFileAsset(root, 'default'), second)
    assert.equal(await removeWallpaperFile(root, 'default'), true)
    assert.equal(await removeWallpaperFile(root, 'default'), false)
    assert.equal(await readWallpaperFileAsset(root, 'default'), null)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('saved wallpaper lookup rejects a symbolic-link replacement', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wallpaper-link-'))

  try {
    const target = path.join(root, 'outside.jpg')
    const assetPath = wallpaperFilePath(root, 'default')

    fs.mkdirSync(path.dirname(assetPath), { recursive: true })
    fs.writeFileSync(target, 'not-an-imported-wallpaper')

    try {
      fs.symlinkSync(target, assetPath, 'file')
    } catch (error) {
      if (['EACCES', 'EPERM'].includes((error as NodeJS.ErrnoException).code ?? '')) {
        return
      }

      throw error
    }

    assert.equal(await readWallpaperFileAsset(root, 'default'), null)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('wallpaper replacement preserves the previous file when the atomic rename fails', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wallpaper-atomic-'))

  try {
    const first = await writeWallpaperFile(root, 'default', Buffer.from('previous'))

    await assert.rejects(
      writeWallpaperFile(root, 'default', Buffer.from('replacement'), {
        rename: async () => {
          throw new Error('simulated rename failure')
        }
      }),
      /simulated rename failure/
    )

    assert.equal(fs.readFileSync(first.filePath, 'utf8'), 'previous')
    assert.deepEqual(fs.readdirSync(path.dirname(first.filePath)), [path.basename(first.filePath)])
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('wallpaper source reads reject a canonical target replaced after validation', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-wallpaper-source-'))

  try {
    const source = path.join(root, 'source.jpg')
    const displaced = path.join(root, 'source.previous.jpg')

    fs.writeFileSync(source, 'trusted-image')

    const expectedStat = fs.statSync(source)

    fs.renameSync(source, displaced)
    fs.writeFileSync(source, 'replacement-image')

    await assert.rejects(readWallpaperSourceFile(source, expectedStat), /changed during validation/)
    assert.equal((await readWallpaperSourceFile(displaced, expectedStat)).toString('utf8'), 'trusted-image')
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})
