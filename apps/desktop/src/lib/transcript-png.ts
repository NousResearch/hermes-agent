// PNG CRC covers the chunk type and data, not the length. This validates framing
// and integrity only; it does not inflate IDAT or prove pixel decodability.
const crcTable = Uint32Array.from({ length: 256 }, (_, n) => {
  let crc = n

  for (let bit = 0; bit < 8; bit++) {
    crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
  }

  return crc >>> 0
})

export function completeTranscriptPng(bytes: Uint8Array): boolean {
  if (bytes.length < 45) {
    return false
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  if (view.getUint32(0) !== 0x89504e47 || view.getUint32(4) !== 0x0d0a1a0a) {
    return false
  }

  let at = 8
  let hasPixels = false

  while (at + 12 <= bytes.length) {
    const length = view.getUint32(at)

    if (length > bytes.length - at - 12) {
      return false
    }

    const type = view.getUint32(at + 4)

    if (at === 8 && (type !== 0x49484452 || length !== 13)) {
      return false
    }

    let crc = 0xffffffff
    const end = at + 8 + length

    for (let i = at + 4; i < end; i++) {
      crc = crcTable[(crc ^ bytes[i]) & 255] ^ (crc >>> 8)
    }

    if (((crc ^ 0xffffffff) >>> 0) !== view.getUint32(end)) {
      return false
    }

    if (type === 0x49444154 && length) {
      hasPixels = true
    }

    at = end + 4

    if (type === 0x49454e44) {
      return length === 0 && at === bytes.length && hasPixels
    }
  }

  return false
}
