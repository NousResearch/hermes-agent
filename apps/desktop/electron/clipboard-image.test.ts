import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

import { readClipboardImage, readClipboardImagePng } from './clipboard-image'

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function fakePngBuffer(extraBytes = 16) {
  return Buffer.concat([PNG_SIGNATURE, Buffer.alloc(extraBytes, 0x42)])
}

function blobOf(buffer) {
  return {
    arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  }
}

function clipboardWith(items) {
  return {
    read: async () => items
  }
}

function itemOf(types, getType) {
  return { types, getType }
}

test('readClipboardImagePng returns PNG bytes from an image/png clipboard item', async () => {
  const png = fakePngBuffer()
  const clipboard = clipboardWith([itemOf(['text/plain', 'image/png'], (type) => {
    assert.equal(type, 'image/png')

    return Promise.resolve(blobOf(png))
  })])

  const result = await readClipboardImagePng(clipboard)

  assert.ok(Buffer.isBuffer(result))
  assert.ok(result.equals(png))
})

test('readClipboardImagePng skips items without an image/png type', async () => {
  const clipboard = clipboardWith([itemOf(['text/plain'], (type) => Promise.reject(new Error(`unexpected ${type}`)))])

  assert.equal(await readClipboardImagePng(clipboard), null)
})

test('readClipboardImagePng keeps scanning when getType rejects for a vanished type', async () => {
  const png = fakePngBuffer()
  const clipboard = clipboardWith([
    itemOf(['image/png'], () => Promise.reject(new Error('type vanished'))),
    itemOf(['image/png'], () => Promise.resolve(blobOf(png)))
  ])

  const result = await readClipboardImagePng(clipboard)

  assert.ok(result && result.equals(png))
})

test('readClipboardImagePng treats an empty PNG payload as no image', async () => {
  const clipboard = clipboardWith([itemOf(['image/png'], () => Promise.resolve(blobOf(Buffer.alloc(0))))])

  assert.equal(await readClipboardImagePng(clipboard), null)
})

test('readClipboardImagePng returns null when clipboard.read() rejects', async () => {
  const clipboard = {
    read: () => Promise.reject(new Error('clipboard busy'))
  }

  assert.equal(await readClipboardImagePng(clipboard), null)
})

test('readClipboardImagePng tolerates malformed items in the read result', async () => {
  const clipboard = clipboardWith([null, { types: 'not-an-array' }, itemOf([], () => Promise.resolve(blobOf(fakePngBuffer())))])

  assert.equal(await readClipboardImagePng(clipboard), null)
})

test('readClipboardImagePng returns null for an empty clipboard', async () => {
  assert.equal(await readClipboardImagePng(clipboardWith([])), null)
})

// macOS Universal Clipboard (an image copied on an iPhone) puts a file URL on
// the pasteboard instead of image bytes; Electron 44 surfaces it as text/uri-list
// and no image/png item (#159184). These drive the helper against real files.
function tempImageFile(name, bytes) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-clipboard-image-'))
  const filePath = path.join(dir, name)

  fs.writeFileSync(filePath, bytes)

  return filePath
}

function uriListClipboard(uriList) {
  return clipboardWith([itemOf(['text/uri-list'], (type) => {
    assert.equal(type, 'text/uri-list')

    return Promise.resolve(new Blob([uriList]))
  })])
}

test('readClipboardImage reads the image file a clipboard file URL names', async () => {
  const png = fakePngBuffer()
  const filePath = tempImageFile('IMG_0001.png', png)

  const image = await readClipboardImage(uriListClipboard(`${pathToFileURL(filePath).href}\r\n`), 1024)

  assert.ok(image)
  assert.ok(image.data.equals(png))
  assert.equal(image.ext, '.png')
  assert.equal(image.name, 'IMG_0001.png')
})

test('readClipboardImage never reads a named file that is not an image or is over the cap', async () => {
  const text = tempImageFile('notes.txt', Buffer.from('secret'))
  const large = tempImageFile('IMG_0002.png', fakePngBuffer(64))

  assert.equal(await readClipboardImage(uriListClipboard(pathToFileURL(text).href), 1024), null)
  assert.equal(await readClipboardImage(uriListClipboard(pathToFileURL(large).href), 32), null)
})
