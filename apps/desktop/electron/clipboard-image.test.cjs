'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const { __testing, readClipboardImagePng } = require('./clipboard-image.cjs')

function nativeImage({ empty = false, png = Buffer.from('png') } = {}) {
  return {
    isEmpty: () => empty,
    toPNG: () => png
  }
}

function fakeClipboard({ image, formats = [], buffers = {} } = {}) {
  return {
    availableFormats: () => formats,
    readBuffer: format => {
      const value = buffers[format]
      if (value instanceof Error) throw value
      return value || Buffer.alloc(0)
    },
    readImage: () => image || nativeImage({ empty: true })
  }
}

function makeDib32BottomUp(width, height, topDownBgraRows) {
  const headerSize = 40
  const rowStride = width * 4
  const buffer = Buffer.alloc(headerSize + rowStride * height)
  buffer.writeUInt32LE(headerSize, 0)
  buffer.writeInt32LE(width, 4)
  buffer.writeInt32LE(height, 8)
  buffer.writeUInt16LE(1, 12)
  buffer.writeUInt16LE(32, 14)
  buffer.writeUInt32LE(0, 16)
  buffer.writeUInt32LE(rowStride * height, 20)

  for (let y = 0; y < height; y += 1) {
    const srcRow = topDownBgraRows[height - 1 - y]
    Buffer.from(srcRow).copy(buffer, headerSize + y * rowStride)
  }

  return buffer
}

function makeDib32TopDown(width, height, topDownBgraRows) {
  const headerSize = 40
  const rowStride = width * 4
  const buffer = Buffer.alloc(headerSize + rowStride * height)
  buffer.writeUInt32LE(headerSize, 0)
  buffer.writeInt32LE(width, 4)
  buffer.writeInt32LE(-height, 8)
  buffer.writeUInt16LE(1, 12)
  buffer.writeUInt16LE(32, 14)
  buffer.writeUInt32LE(0, 16)
  buffer.writeUInt32LE(rowStride * height, 20)

  for (let y = 0; y < height; y += 1) {
    Buffer.from(topDownBgraRows[y]).copy(buffer, headerSize + y * rowStride)
  }

  return buffer
}

function makeDib32Bitfields(width, height, topDownBgraRows) {
  const headerSize = 40
  const maskBytes = 12
  const rowStride = width * 4
  const buffer = Buffer.alloc(headerSize + maskBytes + rowStride * height)
  buffer.writeUInt32LE(headerSize, 0)
  buffer.writeInt32LE(width, 4)
  buffer.writeInt32LE(height, 8)
  buffer.writeUInt16LE(1, 12)
  buffer.writeUInt16LE(32, 14)
  buffer.writeUInt32LE(3, 16) // BI_BITFIELDS
  buffer.writeUInt32LE(rowStride * height, 20)
  buffer.writeUInt32LE(0x00ff0000, headerSize)
  buffer.writeUInt32LE(0x0000ff00, headerSize + 4)
  buffer.writeUInt32LE(0x000000ff, headerSize + 8)

  for (let y = 0; y < height; y += 1) {
    const srcRow = topDownBgraRows[height - 1 - y]
    Buffer.from(srcRow).copy(buffer, headerSize + maskBytes + y * rowStride)
  }

  return buffer
}

function makeDib24BottomUp(width, height, topDownBgrRows) {
  const headerSize = 40
  const rowStride = Math.floor((width * 24 + 31) / 32) * 4
  const buffer = Buffer.alloc(headerSize + rowStride * height)
  buffer.writeUInt32LE(headerSize, 0)
  buffer.writeInt32LE(width, 4)
  buffer.writeInt32LE(height, 8)
  buffer.writeUInt16LE(1, 12)
  buffer.writeUInt16LE(24, 14)
  buffer.writeUInt32LE(0, 16)
  buffer.writeUInt32LE(rowStride * height, 20)

  for (let y = 0; y < height; y += 1) {
    const srcRow = topDownBgrRows[height - 1 - y]
    Buffer.from(srcRow).copy(buffer, headerSize + y * rowStride)
  }

  return buffer
}

test('readClipboardImagePng uses clipboard.readImage first', () => {
  const directPng = Buffer.from('direct')
  const clipboard = fakeClipboard({
    image: nativeImage({ png: directPng }),
    formats: ['image/png'],
    buffers: { 'image/png': Buffer.from('fallback') }
  })

  const result = readClipboardImagePng(clipboard, {
    createFromBuffer: () => nativeImage({ png: Buffer.from('fallback-png') })
  })

  assert.deepEqual(result, directPng)
})

test('readClipboardImagePng falls back to encoded image clipboard buffers', () => {
  const encoded = Buffer.from('encoded-image')
  const converted = Buffer.from('converted-png')
  const clipboard = fakeClipboard({
    formats: ['text/plain', 'image/png'],
    buffers: { 'image/png': encoded }
  })

  const result = readClipboardImagePng(clipboard, {
    createFromBuffer: buffer => {
      assert.deepEqual(buffer, encoded)
      return nativeImage({ png: converted })
    }
  })

  assert.deepEqual(result, converted)
})

test('readClipboardImagePng converts Windows DIB clipboard buffers', () => {
  const topRed = [0, 0, 255, 255]
  const bottomBlue = [255, 0, 0, 255]
  const dib = makeDib32BottomUp(1, 2, [topRed, bottomBlue])
  const png = Buffer.from('dib-png')
  const clipboard = fakeClipboard({
    formats: ['CF_DIB'],
    buffers: { CF_DIB: dib }
  })

  const result = readClipboardImagePng(clipboard, {
    createFromBitmap: (bitmap, options) => {
      assert.deepEqual(options, { width: 1, height: 2 })
      assert.deepEqual([...bitmap], [...topRed, ...bottomBlue])
      return nativeImage({ png })
    },
    createFromBuffer: () => nativeImage({ empty: true })
  })

  assert.deepEqual(result, png)
})

test('readClipboardImagePng probes slash-form BMP when Windows advertises CF_DIB', () => {
  const dib = makeDib32BottomUp(1, 1, [[4, 3, 2, 255]])
  const png = Buffer.from('bmp-dib-png')
  const clipboard = fakeClipboard({
    formats: ['CF_DIB'],
    buffers: {
      CF_DIB: new Error('Electron rejects native CF_DIB format names'),
      'image/bmp': dib
    }
  })

  const result = readClipboardImagePng(clipboard, {
    createFromBitmap: (bitmap, options) => {
      assert.deepEqual(options, { width: 1, height: 1 })
      assert.deepEqual([...bitmap], [4, 3, 2, 255])
      return nativeImage({ png })
    },
    createFromBuffer: () => nativeImage({ empty: true })
  })

  assert.deepEqual(result, png)
})

test('dibToBitmap makes zero-alpha 32-bit DIB pixels visible', () => {
  const dib = makeDib32BottomUp(1, 1, [[8, 7, 6, 0]])

  const parsed = __testing.dibToBitmap(dib)

  assert.deepEqual(parsed, {
    bitmap: Buffer.from([8, 7, 6, 255]),
    width: 1,
    height: 1
  })
})

test('dibToBitmap treats 32-bit BI_RGB alpha bytes as opaque', () => {
  const dib = makeDib32BottomUp(2, 1, [[8, 7, 6, 0, 5, 4, 3, 128]])

  const parsed = __testing.dibToBitmap(dib)

  assert.deepEqual(parsed, {
    bitmap: Buffer.from([8, 7, 6, 255, 5, 4, 3, 255]),
    width: 2,
    height: 1
  })
})

test('dibToBitmap handles 32-bit BI_BITFIELDS masks after BITMAPINFOHEADER', () => {
  const dib = makeDib32Bitfields(1, 1, [[4, 3, 2, 99]])

  const parsed = __testing.dibToBitmap(dib)

  assert.deepEqual(parsed, {
    bitmap: Buffer.from([4, 3, 2, 255]),
    width: 1,
    height: 1
  })
})

test('dibToBitmap handles 24-bit row padding', () => {
  const topRed = [0, 0, 255]
  const bottomBlue = [255, 0, 0]
  const dib = makeDib24BottomUp(1, 2, [topRed, bottomBlue])

  const parsed = __testing.dibToBitmap(dib)

  assert.deepEqual(parsed, {
    bitmap: Buffer.from([0, 0, 255, 255, 255, 0, 0, 255]),
    width: 1,
    height: 2
  })
})

test('dibToBitmap handles top-down negative-height DIBs', () => {
  const first = [1, 2, 3, 255]
  const second = [4, 5, 6, 255]
  const dib = makeDib32TopDown(1, 2, [first, second])

  const parsed = __testing.dibToBitmap(dib)

  assert.deepEqual(parsed, {
    bitmap: Buffer.from([...first, ...second]),
    width: 1,
    height: 2
  })
})

test('readClipboardImagePng returns null when no image format is usable', () => {
  const result = readClipboardImagePng(fakeClipboard({ formats: ['text/plain'] }), {
    createFromBuffer: () => nativeImage({ empty: true }),
    createFromBitmap: () => nativeImage({ empty: true })
  })

  assert.equal(result, null)
})
