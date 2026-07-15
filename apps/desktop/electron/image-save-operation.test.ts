import { describe, expect, it, vi } from 'vitest'

import { createSingleFlightOperation, encodedImageDimensions, imageDimensionsWithinLimit } from './image-save-operation'

function png(width: number, height: number) {
  const buffer = Buffer.alloc(24)
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(buffer)
  buffer.write('IHDR', 12, 'ascii')
  buffer.writeUInt32BE(width, 16)
  buffer.writeUInt32BE(height, 20)

  return buffer
}

function gif(width: number, height: number) {
  const buffer = Buffer.alloc(14)
  buffer.write('GIF89a', 0, 'ascii')
  buffer.writeUInt16LE(width, 6)
  buffer.writeUInt16LE(height, 8)
  buffer[12] = 0
  buffer[13] = 0x3b

  return buffer
}

function bmp(width: number, height: number) {
  const buffer = Buffer.alloc(26)
  buffer.write('BM', 0, 'ascii')
  buffer.writeUInt32LE(40, 14)
  buffer.writeInt32LE(width, 18)
  buffer.writeInt32LE(height, 22)

  return buffer
}

function jpeg(width: number, height: number) {
  const buffer = Buffer.alloc(21)
  buffer[0] = 0xff
  buffer[1] = 0xd8
  buffer[2] = 0xff
  buffer[3] = 0xc0
  buffer.writeUInt16BE(17, 4)
  buffer[6] = 8
  buffer.writeUInt16BE(height, 7)
  buffer.writeUInt16BE(width, 9)
  buffer[11] = 3

  return buffer
}

function webp(width: number, height: number) {
  const buffer = Buffer.alloc(30)
  buffer.write('RIFF', 0, 'ascii')
  buffer.writeUInt32LE(22, 4)
  buffer.write('WEBP', 8, 'ascii')
  buffer.write('VP8X', 12, 'ascii')
  buffer.writeUInt32LE(10, 16)
  buffer.writeUIntLE(width - 1, 24, 3)
  buffer.writeUIntLE(height - 1, 27, 3)

  return buffer
}

function svg(width: number, height: number) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"></svg>`)
}

const formats = [
  ['.png', png],
  ['.jpg', jpeg],
  ['.gif', gif],
  ['.webp', webp],
  ['.bmp', bmp],
  ['.svg', svg]
] as const

describe('createSingleFlightOperation', () => {
  it('rejects overlapping operations and admits a new operation after release', async () => {
    let releaseFirst: () => void = () => undefined

    const firstOperation = new Promise<void>(resolve => {
      releaseFirst = resolve
    })

    const run = createSingleFlightOperation('busy')

    const first = run(() => firstOperation)

    await expect(run(() => Promise.resolve('second'))).rejects.toThrow('busy')

    releaseFirst()
    await first
    await expect(run(() => Promise.resolve('third'))).resolves.toBe('third')
  })

  it('releases the guard when an operation rejects', async () => {
    const run = createSingleFlightOperation()
    const failure = new Error('failed')

    await expect(run(() => Promise.reject(failure))).rejects.toBe(failure)
    await expect(run(() => Promise.resolve(true))).resolves.toBe(true)
  })

  it('invokes only the admitted operation', async () => {
    let releaseFirst: () => void = () => undefined

    const operation = vi.fn(
      () =>
        new Promise<void>(resolve => {
          releaseFirst = resolve
        })
    )

    const rejectedOperation = vi.fn()
    const run = createSingleFlightOperation()

    const first = run(operation)
    await expect(run(rejectedOperation)).rejects.toThrow('already in progress')
    expect(rejectedOperation).not.toHaveBeenCalled()

    releaseFirst()
    await first
  })
})

describe('encodedImageDimensions', () => {
  it.each(formats)('reads encoded dimensions before decoding %s images', (extension, build) => {
    expect(encodedImageDimensions(build(320, 240), extension)).toEqual({ width: 320, height: 240 })
  })

  it.each(formats)('exposes oversized %s headers to the pre-decode policy', (extension, build) => {
    const dimensions = encodedImageDimensions(build(20_000, 10_000), extension)

    expect(dimensions).not.toBeNull()
    expect(imageDimensionsWithinLimit(dimensions!.width, dimensions!.height, 16_384, 40_000_000)).toBe(false)
  })

  it('handles SVG viewBox sizing and unit conversion', () => {
    expect(encodedImageDimensions(Buffer.from('<svg viewBox="0 0 400 200"/>'), '.svg')).toEqual({
      width: 300,
      height: 150
    })
    expect(encodedImageDimensions(Buffer.from('<svg width="2in" height="1in"/>'), '.svg')).toEqual({
      width: 192,
      height: 96
    })
  })

  it('uses the largest GIF frame rather than trusting a small logical screen', () => {
    const buffer = Buffer.alloc(26)
    buffer.write('GIF89a', 0, 'ascii')
    buffer.writeUInt16LE(1, 6)
    buffer.writeUInt16LE(1, 8)
    buffer[13] = 0x2c
    buffer.writeUInt16LE(20_000, 18)
    buffer.writeUInt16LE(10_000, 20)
    buffer[23] = 2
    buffer[24] = 0
    buffer[25] = 0x3b

    expect(encodedImageDimensions(buffer, '.gif')).toEqual({ width: 20_000, height: 10_000 })
  })

  it('ignores fake SVG tags inside XML comments and scans quoted root attributes safely', () => {
    const buffer = Buffer.from(
      '<!-- <svg width="1" height="1"> --><svg data-value=">" width="20000" height="10000"></svg>'
    )

    expect(encodedImageDimensions(buffer, '.svg')).toEqual({ width: 20_000, height: 10_000 })
  })

  it.each([
    '<svg data-width="1" data-height="1" width="20000" height="10000"></svg>',
    `<svg data-note='width="1" height="1"' width="20000" height="10000"></svg>`
  ])('requires exact, quote-aware SVG dimension attributes', source => {
    const dimensions = encodedImageDimensions(Buffer.from(source), '.svg')

    expect(dimensions).not.toBeNull()
    expect(imageDimensionsWithinLimit(dimensions!.width, dimensions!.height, 16_384, 40_000_000)).toBe(false)
  })

  it('does not confuse namespaced attributes with the exact SVG viewBox attribute', () => {
    const source = '<svg foo:viewBox="0 0 1 1" viewBox="0 0 200 100" height="100"></svg>'

    expect(encodedImageDimensions(Buffer.from(source), '.svg')).toEqual({ width: 200, height: 100 })
  })

  it('rejects duplicate exact SVG attributes instead of choosing an ambiguous value', () => {
    expect(encodedImageDimensions(Buffer.from('<svg width="1" width="20000" height="10000"/>'), '.svg')).toBeNull()
  })

  it('rejects adjacent quoted SVG attributes without required whitespace', () => {
    expect(encodedImageDimensions(Buffer.from('<svg width="10"height="20">'), '.svg')).toBeNull()
  })

  it.each([
    '<svg width="10" height="20">',
    "<svg width='10' height='20'/>",
    '<svg width = 10 height = 20 >'
  ])('accepts valid SVG attribute quoting and whitespace: %s', source => {
    expect(encodedImageDimensions(Buffer.from(source), '.svg')).toEqual({ width: 10, height: 20 })
  })

  it.each([
    '<svg width="10 height="20">',
    '<svg width height="20">',
    '<svg width="10" height="20"',
    '<svg foo:width="1" foo:height="1" width="10" height="20">',
    '<svg data-note="viewBox=&quot;0 0 1 1&quot;" viewBox="0 0 20 10" height="10">'
  ])('handles malformed and exact-name SVG attribute cases: %s', source => {
    const dimensions = encodedImageDimensions(Buffer.from(source), '.svg')

    if (source.includes('foo:width')) {
      expect(dimensions).toEqual({ width: 10, height: 20 })
    } else if (source.includes('data-note')) {
      expect(dimensions).toEqual({ width: 20, height: 10 })
    } else {
      expect(dimensions).toBeNull()
    }
  })

  it.each([
    [Buffer.alloc(8), '.png'],
    [Buffer.from([0xff, 0xd8, 0xff, 0xda]), '.jpg'],
    [Buffer.from('<svg width="100%" height="100%"/>'), '.svg'],
    [Buffer.alloc(30), '.webp'],
    [Buffer.alloc(26), '.bmp']
  ])('rejects malformed encoded headers for %s', (buffer, extension) => {
    expect(encodedImageDimensions(buffer, extension)).toBeNull()
  })
})

describe('imageDimensionsWithinLimit', () => {
  it('accepts valid dimensions within both limits', () => {
    expect(imageDimensionsWithinLimit(7680, 4320, 16_384, 40_000_000)).toBe(true)
  })

  it.each([
    [0, 100],
    [100, 0],
    [16_385, 1],
    [1, 16_385],
    [10_000, 5000],
    [Number.NaN, 100],
    [100.5, 100]
  ])('rejects invalid or oversized dimensions %s × %s', (width, height) => {
    expect(imageDimensionsWithinLimit(width, height, 16_384, 40_000_000)).toBe(false)
  })
})
