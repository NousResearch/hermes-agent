import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  BACKGROUND_MAX_FOLDER_IMAGES,
  BackgroundImageRegistry,
  backgroundTokenFromUrl,
  resolveBackgroundImages
} from './background-images'

const tempDirs = new Set<string>()

const tempDir = () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-background-'))
  tempDirs.add(directory)

  return directory
}

afterEach(() => {
  for (const directory of tempDirs) {
    fs.rmSync(directory, { recursive: true, force: true })
  }

  tempDirs.clear()
})

test('resolves a selected image behind an opaque protocol token', async () => {
  const root = tempDir()
  const image = path.join(root, 'wallpaper.png')
  fs.writeFileSync(image, 'png')
  const registry = new BackgroundImageRegistry()

  const result = await resolveBackgroundImages({ kind: 'image', sourcePath: image }, registry)

  assert.equal(result.images.length, 1)
  assert.equal(result.images[0].name, 'wallpaper.png')
  assert.equal(result.images[0].url.includes(image), false)
  const token = backgroundTokenFromUrl(result.images[0].url)
  assert.ok(token)
  assert.equal(registry.resolve(token), fs.realpathSync(image))
})

test('folder scan is top-level, hidden-file safe, sorted, and format filtered', async () => {
  const root = tempDir()
  fs.writeFileSync(path.join(root, 'b.webp'), 'b')
  fs.writeFileSync(path.join(root, 'a.jpg'), 'a')
  fs.writeFileSync(path.join(root, '.secret.png'), 'hidden')
  fs.writeFileSync(path.join(root, 'notes.txt'), 'text')
  fs.mkdirSync(path.join(root, 'nested'))
  fs.writeFileSync(path.join(root, 'nested', 'nested.png'), 'nested')

  const result = await resolveBackgroundImages({ kind: 'folder', sourcePath: root }, new BackgroundImageRegistry())

  assert.deepEqual(
    result.images.map(image => image.name),
    ['a.jpg', 'b.webp']
  )
})

test('folder scan rejects a symlink that escapes the selected directory', async t => {
  const root = tempDir()
  const outside = tempDir()
  const target = path.join(outside, 'outside.png')
  fs.writeFileSync(target, 'png')

  try {
    fs.symlinkSync(target, path.join(root, 'linked.png'))
  } catch (error: any) {
    t.skip(`symlink creation unavailable: ${error.code}`)

    return
  }

  const result = await resolveBackgroundImages({ kind: 'folder', sourcePath: root }, new BackgroundImageRegistry())
  assert.equal(result.error, 'empty')
})

test('folder result is bounded and reports truncation', async () => {
  const root = tempDir()

  for (let index = 0; index < BACKGROUND_MAX_FOLDER_IMAGES + 1; index += 1) {
    fs.writeFileSync(path.join(root, `${String(index).padStart(4, '0')}.png`), 'x')
  }

  const result = await resolveBackgroundImages({ kind: 'folder', sourcePath: root }, new BackgroundImageRegistry())
  assert.equal(result.images.length, BACKGROUND_MAX_FOLDER_IMAGES)
  assert.equal(result.truncated, true)
})

test('registry drops oldest opaque grants after its bound', () => {
  const registry = new BackgroundImageRegistry()
  const firstUrl = registry.authorize('/tmp/first.png')

  for (let index = 0; index < 4_096; index += 1) {
    registry.authorize(`/tmp/${index}.png`)
  }

  const firstToken = backgroundTokenFromUrl(firstUrl)
  assert.ok(firstToken)
  assert.equal(registry.resolve(firstToken), null)
  assert.equal(registry.size, 4_096)
})
