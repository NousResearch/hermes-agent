import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  type ClipboardImageItem,
  type DecodeClipboardImage,
  readClipboardImageAsPng
} from './clipboard-image'

function item(types: readonly string[], values: Record<string, unknown>): ClipboardImageItem {
  return {
    types,
    async getType(type) {
      const value = values[type]

      if (value instanceof Error) {
        throw value
      }

      return value
    }
  }
}

function blob(value: string, type: string): Blob {
  return new Blob([value], { type })
}

function decoder(emptyValues: readonly string[] = []): DecodeClipboardImage {
  return buffer => {
    const value = buffer.toString()

    return {
      isEmpty: () => emptyValues.includes(value),
      toPNG: () => Buffer.from(`png:${value}`)
    }
  }
}

test('reads PNG clipboard data', async () => {
  const result = await readClipboardImageAsPng(
    [item(['image/png'], { 'image/png': blob('source-png', 'image/png') })],
    decoder()
  )

  assert.equal(result?.toString(), 'png:source-png')
})

test('falls back to JPEG when PNG is unavailable', async () => {
  const result = await readClipboardImageAsPng(
    [item(['image/jpeg'], { 'image/jpeg': blob('source-jpeg', 'image/jpeg') })],
    decoder()
  )

  assert.equal(result?.toString(), 'png:source-jpeg')
})

test('ignores unsupported image types even when they precede PNG', async () => {
  const result = await readClipboardImageAsPng(
    [
      item(['image/svg+xml', 'image/png'], {
        'image/svg+xml': blob('<svg/>', 'image/svg+xml'),
        'image/png': blob('source-png', 'image/png')
      })
    ],
    decoder()
  )

  assert.equal(result?.toString(), 'png:source-png')
})

test('continues through decode failures, invalid values, rejected representations, and later items', async () => {
  const decode = decoder(['empty'])

  const result = await readClipboardImageAsPng(
    [
      item(['image/png', 'image/jpeg'], {
        'image/png': blob('throws', 'image/png'),
        'image/jpeg': new Error('clipboard representation disappeared')
      }),
      item(['image/png'], { 'image/png': 'not-a-blob' }),
      item(['image/png'], { 'image/png': blob('empty', 'image/png') }),
      item(['image/png'], { 'image/png': blob('valid', 'image/png') })
    ],
    buffer => {
      if (buffer.toString() === 'throws') {
        throw new Error('decoder rejected image')
      }

      return decode(buffer)
    }
  )

  assert.equal(result?.toString(), 'png:valid')
})

test('returns null when the clipboard has no decodable PNG or JPEG', async () => {
  const result = await readClipboardImageAsPng(
    [item(['text/plain', 'image/webp'], { 'image/webp': blob('webp', 'image/webp') })],
    decoder()
  )

  assert.equal(result, null)
})