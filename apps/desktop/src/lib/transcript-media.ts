import { completeTranscriptPng } from './transcript-png'

export const TRANSCRIPT_IMAGE_MAX_BYTES = 16 * 1024 * 1024
export const TRANSCRIPT_IMAGE_MAX_BASE64 = 4 * Math.ceil(TRANSCRIPT_IMAGE_MAX_BYTES / 3)

export interface TranscriptImage {
  bytes: Uint8Array
  mime: string
  extension: string
  width?: number
  height?: number
}

export function escapeTranscriptXml(value: string): string {
  return value
    .replace(/\p{Cc}/gu, char => (char.charCodeAt(0) < 32 && !['\t', '\n', '\r'].includes(char) ? '' : char))
    .replace(/[\uFFFE\uFFFF]/g, '')
    .replace(/\p{Surrogate}/gu, '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

export function transcriptBase64(bytes: Uint8Array): string {
  let binary = ''

  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192))
  }

  return btoa(binary)
}

function jpegSize(bytes: Uint8Array): [number, number] | undefined {
  let at = 2

  while (at + 8 < bytes.length) {
    if (bytes[at++] !== 255) {
      return undefined
    }

    while (bytes[at] === 255) {
      at++
    }

    const marker = bytes[at++]

    if (marker === 0xd9 || marker === 0xda) {
      return undefined
    }

    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue
    }

    const size = (bytes[at] << 8) | bytes[at + 1]

    if (size < 2 || at + size > bytes.length) {
      return undefined
    }

    if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
      return [(bytes[at + 5] << 8) | bytes[at + 6], (bytes[at + 3] << 8) | bytes[at + 4]]
    }

    at += size
  }

  return undefined
}

// Preserve ODT's existing baseline/progressive JPEG and scan-header requirements.
function odtJpegSize(bytes: Uint8Array): number[] | undefined {
  if (bytes.length < 32 || bytes[0] !== 255 || bytes[1] !== 216 ||
    bytes[bytes.length - 2] !== 255 || bytes[bytes.length - 1] !== 217) { return undefined }

  let p = 2,
    hasFrame = false,
    hasScan = false,
    width = 0,
    height = 0

  while (p + 4 < bytes.length && bytes[p] === 255) {
    const marker = bytes[p + 1],
      len = bytes[p + 2] * 256 + bytes[p + 3]

    if (len < 2 || p + 2 + len > bytes.length) { break }
    if ([0xc0, 0xc1, 0xc2].includes(marker) && len >= 8) {
      height = bytes[p + 5] * 256 + bytes[p + 6]
      width = bytes[p + 7] * 256 + bytes[p + 8]
      hasFrame = true
    }
    if (marker === 0xda) {
      hasScan = true
      break
    }
    p += 2 + len
  }

  return hasFrame && hasScan ? [width, height] : undefined
}

function pngSize(bytes: Uint8Array): number[] | undefined {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  if (
    bytes.length < 33 ||
    view.getUint32(0) !== 0x89504e47 ||
    view.getUint32(4) !== 0x0d0a1a0a ||
    view.getUint32(12) !== 0x49484452 ||
    !completeTranscriptPng(bytes)
  ) {
    return undefined
  }

  return [view.getUint32(16), view.getUint32(20)]
}

function checkedJpegSize(bytes: Uint8Array): number[] | undefined {
  if (
    bytes.length < 32 ||
    bytes[0] !== 255 ||
    bytes[1] !== 216 ||
    bytes[2] !== 255 ||
    bytes[bytes.length - 2] !== 255 ||
    bytes[bytes.length - 1] !== 217
  ) {
    return undefined
  }

  return jpegSize(bytes)
}

function gifSize(bytes: Uint8Array): number[] | undefined {
  const header = String.fromCharCode(...bytes.subarray(0, 6))

  if (
    bytes.length < 35 ||
    (header !== 'GIF87a' && header !== 'GIF89a') ||
    bytes[bytes.length - 1] !== 59 ||
    !bytes.includes(0x2c, 13)
  ) {
    return undefined
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  return [view.getUint16(6, true), view.getUint16(8, true)]
}

function webpSize(bytes: Uint8Array): number[] | undefined {
  if (bytes.length < 30) {
    return undefined
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  if (
    view.getUint32(0) !== 0x52494646 ||
    view.getUint32(8) !== 0x57454250 ||
    view.getUint32(4, true) + 8 !== bytes.length ||
    view.getUint32(16, true) + 20 > bytes.length
  ) {
    return undefined
  }

  const chunk = String.fromCharCode(...bytes.subarray(12, 16))

  if (chunk === 'VP8X') {
    return [1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16), 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16)]
  }

  if (chunk === 'VP8L' && bytes[20] === 0x2f) {
    const size = view.getUint32(21, true)

    return [(size & 0x3fff) + 1, ((size >>> 14) & 0x3fff) + 1]
  }

  if (chunk === 'VP8 ' && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) {
    return [view.getUint16(26, true) & 0x3fff, view.getUint16(28, true) & 0x3fff]
  }

  return undefined
}

const IMAGE_SIZES: Record<string, (bytes: Uint8Array) => number[] | undefined> = {
  png: pngSize,
  jpeg: checkedJpegSize,
  gif: gifSize,
  webp: webpSize
}

/** Read only bounded inline raster bytes. Never fetch a URL or open a local path. */
export function transcriptImage(value?: string, format?: 'odt'): TranscriptImage | undefined {
  if (!value || value.length > TRANSCRIPT_IMAGE_MAX_BASE64 + 64) {
    return undefined
  }

  const match = /^data:(image\/(png|jpeg|gif|webp));base64,([A-Za-z0-9+/]+={0,2})$/.exec(value)

  if (!match || match[3].length % 4 !== 0) {
    return undefined
  }

  let bytes: Uint8Array

  try {
    bytes = Uint8Array.from(atob(match[3]), c => c.charCodeAt(0))
  } catch {
    return undefined
  }

  if (bytes.length > TRANSCRIPT_IMAGE_MAX_BYTES || bytes.length < 12 || transcriptBase64(bytes) !== match[3]) {
    return undefined
  }

  const size = format === 'odt' && match[2] === 'jpeg' ? odtJpegSize(bytes) : IMAGE_SIZES[match[2]](bytes)

  if (size === undefined) {
    return undefined
  }

  const maxDimension = format === 'odt' ? 8192 : 16384

  if (size.length && (!size[0] || !size[1] || size[0] > maxDimension || size[1] > maxDimension || size[0] * size[1] > 16_000_000)) {
    return undefined
  }

  return {
    bytes,
    mime: match[1],
    extension: match[2] === 'jpeg' ? 'jpg' : match[2],
    width: size?.[0],
    height: size?.[1]
  }
}
