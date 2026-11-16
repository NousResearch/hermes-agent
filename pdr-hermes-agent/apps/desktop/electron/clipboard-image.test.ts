import assert from 'node:assert/strict'

import { test } from 'vitest'

import { readClipboardImagePng } from './clipboard-image'

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
