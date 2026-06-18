'use strict'

const ENCODED_IMAGE_FORMATS = [
  'image/png',
  'image/jpeg',
  'image/jpg',
  'image/bmp',
  'image/x-bmp',
  'application/x-ms-bmp'
]
const DIB_READ_FORMATS = ['image/x-dib', 'application/dib']
const FALLBACK_READ_FORMATS = [...ENCODED_IMAGE_FORMATS, ...DIB_READ_FORMATS]

function imageToPngBuffer(image) {
  if (!image || typeof image.isEmpty !== 'function' || image.isEmpty()) {
    return null
  }
  if (typeof image.toPNG !== 'function') {
    return null
  }
  const png = image.toPNG()
  return Buffer.isBuffer(png) && png.length > 0 ? png : null
}

function safeReadBuffer(clipboardImpl, format) {
  try {
    const buffer = clipboardImpl.readBuffer(format)
    return Buffer.isBuffer(buffer) && buffer.length > 0 ? buffer : null
  } catch {
    return null
  }
}

function safeReadImage(clipboardImpl) {
  try {
    return clipboardImpl.readImage?.('clipboard')
  } catch {
    return null
  }
}

function safeCreateFromBuffer(nativeImageImpl, buffer) {
  try {
    return nativeImageImpl.createFromBuffer(buffer)
  } catch {
    return null
  }
}

function safeCreateFromBitmap(nativeImageImpl, bitmap, options) {
  try {
    return nativeImageImpl.createFromBitmap(bitmap, options)
  } catch {
    return null
  }
}

function clipboardFormats(clipboardImpl) {
  try {
    const formats = clipboardImpl.availableFormats('clipboard')
    return Array.isArray(formats) ? formats.map(String) : []
  } catch {
    return []
  }
}

function orderedImageFormats(formats) {
  const seen = new Set()
  const ordered = []
  for (const format of FALLBACK_READ_FORMATS) {
    const match = formats.find(item => item.toLowerCase() === format)
    if (match && !seen.has(format)) {
      seen.add(format)
      ordered.push(match)
    } else if (!seen.has(format)) {
      seen.add(format)
      ordered.push(format)
    }
  }
  for (const format of formats) {
    const lower = format.toLowerCase()
    if (seen.has(lower)) continue
    if (lower.startsWith('image/') || lower.includes('bitmap') || lower.includes('dib')) {
      seen.add(lower)
      ordered.push(format)
    }
  }
  return ordered
}

function formatMayBeDib(format) {
  const lower = String(format || '').toLowerCase()
  return (
    lower === 'cf_dib' ||
    lower === 'cf_dibv5' ||
    lower === 'image/bmp' ||
    lower === 'image/x-bmp' ||
    lower === 'image/x-dib' ||
    lower === 'application/dib' ||
    lower === 'application/x-ms-bmp' ||
    lower.includes('dib')
  )
}

function pixelOffsetForDib(buffer, headerSize, bitCount, compression) {
  let offset = headerSize

  // BITMAPINFOHEADER + BI_BITFIELDS stores channel masks immediately after
  // the 40-byte header. BITMAPV4/V5 already include those masks in headerSize.
  if (headerSize === 40 && compression === 3) {
    offset += 12
  }

  if (bitCount <= 8) {
    const colorsUsed = buffer.length >= 40 ? buffer.readUInt32LE(32) : 0
    offset += (colorsUsed || (1 << bitCount)) * 4
  }

  return offset
}

function dibHasExplicitAlphaMask(buffer, headerSize, bitCount) {
  if (bitCount !== 32 || headerSize < 56 || buffer.length < 56) {
    return false
  }
  return buffer.readUInt32LE(52) !== 0
}

function dibToBitmap(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 40) {
    return null
  }

  const headerSize = buffer.readUInt32LE(0)
  if (headerSize < 40 || headerSize > buffer.length) {
    return null
  }

  const width = buffer.readInt32LE(4)
  const signedHeight = buffer.readInt32LE(8)
  const planes = buffer.readUInt16LE(12)
  const bitCount = buffer.readUInt16LE(14)
  const compression = buffer.readUInt32LE(16)
  const height = Math.abs(signedHeight)

  if (width <= 0 || height <= 0 || planes !== 1) {
    return null
  }
  if (bitCount !== 24 && bitCount !== 32) {
    return null
  }
  if (compression !== 0 && compression !== 3) {
    return null
  }

  const rowStride = Math.floor((width * bitCount + 31) / 32) * 4
  const pixelOffset = pixelOffsetForDib(buffer, headerSize, bitCount, compression)
  const required = pixelOffset + rowStride * height
  if (pixelOffset < headerSize || required > buffer.length) {
    return null
  }

  const topDown = signedHeight < 0
  const bitmap = Buffer.alloc(width * height * 4)
  const preserveAlpha = dibHasExplicitAlphaMask(buffer, headerSize, bitCount)
  let sawAlpha = false

  for (let y = 0; y < height; y += 1) {
    const srcY = topDown ? y : height - 1 - y
    const srcRow = pixelOffset + srcY * rowStride
    const dstRow = y * width * 4

    for (let x = 0; x < width; x += 1) {
      const src = srcRow + x * (bitCount / 8)
      const dst = dstRow + x * 4
      bitmap[dst] = buffer[src]
      bitmap[dst + 1] = buffer[src + 1]
      bitmap[dst + 2] = buffer[src + 2]
      bitmap[dst + 3] = preserveAlpha ? buffer[src + 3] : 255
      if (preserveAlpha && bitmap[dst + 3] !== 0) {
        sawAlpha = true
      }
    }
  }

  // Many Windows DIB producers leave the alpha byte zero even though pixels
  // are opaque. Preserve explicit real alpha when present, otherwise make it
  // visible.
  if (preserveAlpha && !sawAlpha) {
    for (let i = 3; i < bitmap.length; i += 4) {
      bitmap[i] = 255
    }
  }

  return { bitmap, width, height }
}

function imageFromDib(nativeImageImpl, buffer) {
  const parsed = dibToBitmap(buffer)
  if (!parsed) {
    return null
  }
  return safeCreateFromBitmap(nativeImageImpl, parsed.bitmap, {
    width: parsed.width,
    height: parsed.height
  })
}

function readClipboardImagePng(clipboardImpl, nativeImageImpl) {
  const direct = imageToPngBuffer(safeReadImage(clipboardImpl))
  if (direct) {
    return direct
  }

  const formats = orderedImageFormats(clipboardFormats(clipboardImpl))
  for (const format of formats) {
    const buffer = safeReadBuffer(clipboardImpl, format)
    if (!buffer) continue

    const encoded = imageToPngBuffer(safeCreateFromBuffer(nativeImageImpl, buffer))
    if (encoded) {
      return encoded
    }

    if (formatMayBeDib(format)) {
      const dib = imageToPngBuffer(imageFromDib(nativeImageImpl, buffer))
      if (dib) {
        return dib
      }
    }
  }

  return null
}

module.exports = {
  __testing: { dibToBitmap, orderedImageFormats },
  readClipboardImagePng
}
