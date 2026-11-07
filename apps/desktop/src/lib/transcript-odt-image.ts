import { TRANSCRIPT_IMAGE_MAX_BASE64, TRANSCRIPT_IMAGE_MAX_BYTES } from './transcript-media'
import { completeTranscriptPng } from './transcript-png'

interface Dimensions {
  width: number
  height: number
}
export interface OdtImage extends Dimensions {
  bytes: Uint8Array
  type: string
  ext: string
}

const signature = (bytes: Uint8Array, start: number, n: number) =>
  String.fromCharCode(...bytes.subarray(start, start + n))

function png(bytes: Uint8Array): Dimensions | undefined {
  if (!completeTranscriptPng(bytes)) {
    return
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  return { width: view.getUint32(16), height: view.getUint32(20) }
}

function jpeg(bytes: Uint8Array): Dimensions | undefined {
  if (
    bytes.length < 32 ||
    bytes[0] !== 255 ||
    bytes[1] !== 216 ||
    bytes[bytes.length - 2] !== 255 ||
    bytes[bytes.length - 1] !== 217
  ) {
    return
  }

  let p = 2,
    hasFrame = false,
    hasScan = false,
    width = 0,
    height = 0

  while (p + 4 < bytes.length && bytes[p] === 255) {
    const marker = bytes[p + 1],
      len = bytes[p + 2] * 256 + bytes[p + 3]

    if (len < 2 || p + 2 + len > bytes.length) {
      break
    }

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

  if (hasFrame && hasScan) {
    return { width, height }
  }
}

function gif(bytes: Uint8Array): Dimensions | undefined {
  if (
    bytes.length < 35 ||
    !/^GIF8[79]a$/.test(signature(bytes, 0, 6)) ||
    bytes[bytes.length - 1] !== 59 ||
    !bytes.includes(0x2c, 13)
  ) {
    return
  }

  return { width: bytes[6] + 256 * bytes[7], height: bytes[8] + 256 * bytes[9] }
}

function webp(bytes: Uint8Array): Dimensions | undefined {
  if (bytes.length < 30 || signature(bytes, 0, 4) !== 'RIFF' || signature(bytes, 8, 4) !== 'WEBP') {
    return
  }

  const view = new DataView(bytes.buffer),
    chunk = signature(bytes, 12, 4),
    size = view.getUint32(16, true)

  if (view.getUint32(4, true) + 8 !== bytes.length || size + 20 > bytes.length) {
    return
  }

  if (chunk === 'VP8 ' && signature(bytes, 23, 3) === '\x9d\x01\x2a') {
    return { width: view.getUint16(26, true) & 0x3fff, height: view.getUint16(28, true) & 0x3fff }
  }

  if (chunk === 'VP8L' && bytes[20] === 0x2f && bytes.length >= 25) {
    const n = view.getUint32(21, true)

    return { width: (n & 0x3fff) + 1, height: ((n >>> 14) & 0x3fff) + 1 }
  }

  if (chunk === 'VP8X' && bytes.length >= 30) {
    return {
      width: 1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16),
      height: 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16)
    }
  }
}

const formats: Record<string, (bytes: Uint8Array) => Dimensions | undefined> = { png, jpeg, gif, webp }

export function image(data: string): OdtImage | undefined {
  if (data.length > TRANSCRIPT_IMAGE_MAX_BASE64 + 64) {
    return
  }

  const m = /^data:image\/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(data)

  if (!m || m[2].length > TRANSCRIPT_IMAGE_MAX_BASE64 || m[2].length % 4) {
    return
  }

  try {
    const raw = atob(m[2])

    if (raw.length > TRANSCRIPT_IMAGE_MAX_BYTES || btoa(raw) !== m[2]) {
      return
    }

    const bytes = Uint8Array.from(raw, c => c.charCodeAt(0))
    const dimensions = formats[m[1]](bytes)

    if (!dimensions) {
      return
    }

    const { width, height } = dimensions

    if (!width || !height || width > 8192 || height > 8192 || width * height > 16000000) {
      return
    }

    return { bytes, type: `image/${m[1]}`, ext: m[1] === 'jpeg' ? 'jpg' : m[1], width, height }
  } catch {
    return
  }
}
