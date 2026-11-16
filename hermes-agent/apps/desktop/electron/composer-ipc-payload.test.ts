import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  COMPOSER_IMAGE_MAX_BYTES,
  COMPOSER_PASTE_MAX_BYTES,
  composerImageBufferFromIpc,
  composerPasteTextFromIpc
} from './hardening'

const codeOf = (code: string) => (error: unknown) => (error as { code?: string })?.code === code

test('composerImageBufferFromIpc accepts the byte shapes the renderer sends', () => {
  const bytes = Uint8Array.from([137, 80, 78, 71])

  assert.deepEqual([...composerImageBufferFromIpc(bytes)], [137, 80, 78, 71])
  assert.deepEqual([...composerImageBufferFromIpc(Buffer.from(bytes))], [137, 80, 78, 71])
  assert.deepEqual([...composerImageBufferFromIpc(bytes.buffer)], [137, 80, 78, 71])
})

test('composerImageBufferFromIpc rejects missing, empty and non-byte payloads', () => {
  for (const data of [undefined, null, new Uint8Array(0), 'not bytes', { length: 1e12 }]) {
    assert.throws(() => composerImageBufferFromIpc(data), codeOf('invalid-image-data'))
  }
})

test('composerImageBufferFromIpc rejects a payload over the image cap', () => {
  assert.throws(() => composerImageBufferFromIpc(new Uint8Array(5), 4), codeOf('EFBIG'))
  assert.equal(composerImageBufferFromIpc(new Uint8Array(4), 4).byteLength, 4)
  // A view over a large buffer is sized by its own window, not the backing store.
  const big = new Uint8Array(COMPOSER_IMAGE_MAX_BYTES + 1)
  assert.throws(() => composerImageBufferFromIpc(big), codeOf('EFBIG'))
  assert.equal(composerImageBufferFromIpc(big.subarray(0, 8)).byteLength, 8)
})

test('composerPasteTextFromIpc caps the pasted text by its UTF-8 size', () => {
  assert.equal(composerPasteTextFromIpc('hello'), 'hello')
  assert.throws(() => composerPasteTextFromIpc(''), codeOf('invalid-text'))
  assert.throws(() => composerPasteTextFromIpc(42), codeOf('invalid-text'))
  assert.throws(() => composerPasteTextFromIpc('é'.repeat(3), 5), codeOf('EFBIG'))
  assert.throws(() => composerPasteTextFromIpc('a'.repeat(COMPOSER_PASTE_MAX_BYTES + 1)), codeOf('EFBIG'))
})
