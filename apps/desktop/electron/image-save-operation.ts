export type ImageDimensions = { width: number; height: number }

export function createSingleFlightOperation(errorMessage = 'Another operation is already in progress') {
  let active = false

  return async function runSingleFlight<T>(operation: () => Promise<T> | T): Promise<T> {
    if (active) {
      throw new Error(errorMessage)
    }

    active = true

    try {
      return await operation()
    } finally {
      active = false
    }
  }
}

export function imageDimensionsWithinLimit(
  width: number,
  height: number,
  maxDimension: number,
  maxPixels: number
): boolean {
  return (
    Number.isSafeInteger(width) &&
    Number.isSafeInteger(height) &&
    width > 0 &&
    height > 0 &&
    width <= maxDimension &&
    height <= maxDimension &&
    width * height <= maxPixels
  )
}

function pngDimensions(buffer: Buffer): ImageDimensions | null {
  if (buffer.length < 24 || buffer.subarray(12, 16).toString('ascii') !== 'IHDR') {
    return null
  }

  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) }
}

function gifDimensions(buffer: Buffer): ImageDimensions | null {
  if (buffer.length < 14) {
    return null
  }

  const screenWidth = buffer.readUInt16LE(6)
  const screenHeight = buffer.readUInt16LE(8)
  const packed = buffer[10]
  const globalColorTableBytes = packed & 0x80 ? 3 * 2 ** ((packed & 0x07) + 1) : 0
  let offset = 13 + globalColorTableBytes
  let maxWidth = screenWidth
  let maxHeight = screenHeight

  const skipSubBlocks = () => {
    while (offset < buffer.length) {
      const byteLength = buffer[offset]
      offset += 1

      if (byteLength === 0) {
        return true
      }

      if (offset + byteLength > buffer.length) {
        return false
      }

      offset += byteLength
    }

    return false
  }

  while (offset < buffer.length) {
    const blockType = buffer[offset]
    offset += 1

    if (blockType === 0x3b) {
      return { width: maxWidth, height: maxHeight }
    }

    if (blockType === 0x21) {
      if (offset >= buffer.length) {
        return null
      }

      offset += 1

      if (!skipSubBlocks()) {
        return null
      }

      continue
    }

    if (blockType !== 0x2c || offset + 9 > buffer.length) {
      return null
    }

    const frameWidth = buffer.readUInt16LE(offset + 4)
    const frameHeight = buffer.readUInt16LE(offset + 6)
    const framePacked = buffer[offset + 8]
    offset += 9
    maxWidth = Math.max(maxWidth, frameWidth)
    maxHeight = Math.max(maxHeight, frameHeight)

    if (framePacked & 0x80) {
      offset += 3 * 2 ** ((framePacked & 0x07) + 1)
    }

    if (offset >= buffer.length) {
      return null
    }

    offset += 1

    if (!skipSubBlocks()) {
      return null
    }
  }

  return null
}

function bmpDimensions(buffer: Buffer): ImageDimensions | null {
  if (buffer.length < 26) {
    return null
  }

  const dibHeaderSize = buffer.readUInt32LE(14)

  if (dibHeaderSize === 12) {
    return { width: buffer.readUInt16LE(18), height: buffer.readUInt16LE(20) }
  }

  if (dibHeaderSize < 40) {
    return null
  }

  return { width: Math.abs(buffer.readInt32LE(18)), height: Math.abs(buffer.readInt32LE(22)) }
}

const JPEG_START_OF_FRAME = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf])

function jpegDimensions(buffer: Buffer): ImageDimensions | null {
  if (buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) {
    return null
  }

  let offset = 2
  let dimensions: ImageDimensions | null = null

  while (offset < buffer.length) {
    if (buffer[offset] !== 0xff) {
      return null
    }

    while (offset < buffer.length && buffer[offset] === 0xff) {
      offset += 1
    }

    if (offset >= buffer.length) {
      return null
    }

    const marker = buffer[offset]
    offset += 1

    if (marker === 0xd9 || marker === 0xda) {
      return dimensions
    }

    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue
    }

    if (offset + 2 > buffer.length) {
      return null
    }

    const segmentLength = buffer.readUInt16BE(offset)

    if (segmentLength < 2 || offset + segmentLength > buffer.length) {
      return null
    }

    if (JPEG_START_OF_FRAME.has(marker)) {
      if (segmentLength < 7 || dimensions) {
        return null
      }

      dimensions = { width: buffer.readUInt16BE(offset + 5), height: buffer.readUInt16BE(offset + 3) }
    }

    offset += segmentLength
  }

  return dimensions
}

function readUInt24LE(buffer: Buffer, offset: number): number {
  return buffer[offset] | (buffer[offset + 1] << 8) | (buffer[offset + 2] << 16)
}

function webpDimensions(buffer: Buffer): ImageDimensions | null {
  if (buffer.length < 30) {
    return null
  }

  const chunkType = buffer.subarray(12, 16).toString('ascii')

  if (chunkType === 'VP8X') {
    return { width: readUInt24LE(buffer, 24) + 1, height: readUInt24LE(buffer, 27) + 1 }
  }

  if (chunkType === 'VP8 ' && buffer[23] === 0x9d && buffer[24] === 0x01 && buffer[25] === 0x2a) {
    return { width: buffer.readUInt16LE(26) & 0x3fff, height: buffer.readUInt16LE(28) & 0x3fff }
  }

  if (chunkType === 'VP8L' && buffer[20] === 0x2f) {
    const bits = buffer.readUInt32LE(21)

    return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 }
  }

  return null
}

function svgAttributes(tag: string): Map<string, string> | null {
  const attributes = new Map<string, string>()
  let index = 4

  while (index < tag.length) {
    while (/\s/.test(tag[index] || '')) {
      index += 1
    }

    if (tag[index] === '>' || (tag[index] === '/' && tag[index + 1] === '>')) {
      return attributes
    }

    const nameStart = index

    while (index < tag.length && !/[\s=/>]/.test(tag[index])) {
      index += 1
    }

    const name = tag.slice(nameStart, index)

    if (!name || attributes.has(name)) {
      return null
    }

    while (/\s/.test(tag[index] || '')) {
      index += 1
    }

    if (tag[index] !== '=') {
      return null
    }

    index += 1

    while (/\s/.test(tag[index] || '')) {
      index += 1
    }

    const quote = tag[index] === '"' || tag[index] === "'" ? tag[index] : ''
    let value = ''

    if (quote) {
      index += 1
      const valueStart = index

      while (index < tag.length && tag[index] !== quote) {
        index += 1
      }

      if (index >= tag.length) {
        return null
      }

      value = tag.slice(valueStart, index)
      index += 1

      if (
        index < tag.length &&
        !/\s/.test(tag[index]) &&
        tag[index] !== '>' &&
        !(tag[index] === '/' && tag[index + 1] === '>')
      ) {
        return null
      }
    } else {
      const valueStart = index

      while (index < tag.length && !/[\s/>]/.test(tag[index])) {
        index += 1
      }

      value = tag.slice(valueStart, index)

      if (!value) {
        return null
      }
    }

    attributes.set(name, value)
  }

  return null
}

function svgLength(value: string): number | null {
  const match = value.match(/^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(px|pt|pc|in|cm|mm|q)?\s*$/i)

  if (!match) {
    return null
  }

  const amount = Number(match[1])
  const unit = (match[2] || 'px').toLowerCase()
  const factors: Record<string, number> = {
    px: 1,
    pt: 96 / 72,
    pc: 16,
    in: 96,
    cm: 96 / 2.54,
    mm: 96 / 25.4,
    q: 96 / 101.6
  }
  const pixels = amount * factors[unit]

  return Number.isFinite(pixels) && pixels > 0 ? pixels : null
}

function svgDimensions(buffer: Buffer): ImageDimensions | null {
  const text = buffer
    .subarray(0, 64 * 1024)
    .toString('utf8')
    .replace(/^\uFEFF/, '')
    .trimStart()
    .replace(/^(?:<\?xml[^>]*>\s*)?(?:<!--[\s\S]*?-->\s*)*/i, '')

  if (!/^<svg\b/i.test(text)) {
    return null
  }

  let quote = ''
  let tagEnd = -1

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]

    if (quote) {
      if (character === quote) {
        quote = ''
      }
    } else if (character === '"' || character === "'") {
      quote = character
    } else if (character === '>') {
      tagEnd = index
      break
    }
  }

  const tag = tagEnd >= 0 ? text.slice(0, tagEnd + 1) : null

  if (!tag) {
    return null
  }

  const attributes = svgAttributes(tag)

  if (!attributes) {
    return null
  }

  const rawWidth = attributes.get('width') ?? null
  const rawHeight = attributes.get('height') ?? null
  const width = rawWidth === null ? null : svgLength(rawWidth)
  const height = rawHeight === null ? null : svgLength(rawHeight)

  if ((rawWidth !== null && width === null) || (rawHeight !== null && height === null)) {
    return null
  }

  const viewBox = attributes
    .get('viewBox')
    ?.trim()
    .split(/[\s,]+/)
    .map(Number)
  const viewBoxWidth = viewBox?.length === 4 && Number.isFinite(viewBox[2]) && viewBox[2] > 0 ? viewBox[2] : null
  const viewBoxHeight = viewBox?.length === 4 && Number.isFinite(viewBox[3]) && viewBox[3] > 0 ? viewBox[3] : null

  let resolvedWidth = width
  let resolvedHeight = height

  if (resolvedWidth === null && resolvedHeight !== null && viewBoxWidth && viewBoxHeight) {
    resolvedWidth = (resolvedHeight * viewBoxWidth) / viewBoxHeight
  }

  if (resolvedHeight === null && resolvedWidth !== null && viewBoxWidth && viewBoxHeight) {
    resolvedHeight = (resolvedWidth * viewBoxHeight) / viewBoxWidth
  }

  resolvedWidth ??= 300
  resolvedHeight ??= 150

  if (!Number.isFinite(resolvedWidth) || !Number.isFinite(resolvedHeight)) {
    return null
  }

  return { width: Math.ceil(resolvedWidth), height: Math.ceil(resolvedHeight) }
}

export function encodedImageDimensions(buffer: Buffer, extension: string): ImageDimensions | null {
  switch (extension.toLowerCase()) {
    case '.png':
      return pngDimensions(buffer)
    case '.jpg':
    case '.jpeg':
      return jpegDimensions(buffer)
    case '.gif':
      return gifDimensions(buffer)
    case '.webp':
      return webpDimensions(buffer)
    case '.bmp':
      return bmpDimensions(buffer)
    case '.svg':
      return svgDimensions(buffer)
    default:
      return null
  }
}
