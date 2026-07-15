import { randomBytes } from 'node:crypto'

const MAX_FILENAME_LENGTH = 240
const MAX_FILENAME_INPUT_LENGTH = 4096
const BIDI_CONTROL = /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/
const WINDOWS_DEVICE_NAME = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i

function boundFilename(filename: string): string {
  if (Buffer.byteLength(filename, 'utf8') <= MAX_FILENAME_LENGTH) {
    return filename
  }

  const extension = filename.match(/\.[a-z0-9]{2,5}$/i)?.[0] || ''
  const stem = extension ? filename.slice(0, -extension.length) : filename
  const stemByteLimit = Math.max(1, MAX_FILENAME_LENGTH - Buffer.byteLength(extension, 'utf8'))
  let boundedStem = ''
  let byteLength = 0

  for (const character of stem) {
    const characterBytes = Buffer.byteLength(character, 'utf8')

    if (byteLength + characterBytes > stemByteLimit) {
      break
    }

    boundedStem += character
    byteLength += characterBytes
  }

  return `${boundedStem}${extension}`
}

function safeBasename(value?: string): string {
  const raw = [...String(value || '')].slice(0, MAX_FILENAME_INPUT_LENGTH).join('').trim()

  if (!raw) {
    return ''
  }

  let decoded = raw

  try {
    decoded = decodeURIComponent(raw)
  } catch {
    // Keep the original text when it contains malformed percent escapes.
  }

  const base = decoded.split(/[\\/]/).filter(Boolean).pop() || ''

  const printable = [...base]
    .filter(
      character =>
        character.charCodeAt(0) >= 32 &&
        character.charCodeAt(0) !== 127 &&
        !(character.charCodeAt(0) >= 128 && character.charCodeAt(0) <= 159) &&
        !BIDI_CONTROL.test(character)
    )
    .join('')

  let safe = printable
    .replace(/[<>:"/\\|?*]/g, '-')
    .replace(/^[. ]+/, '')
    .replace(/[. ]+$/g, '')
    .trim()

  if (!safe || /^\.+$/.test(safe)) {
    return ''
  }

  if (WINDOWS_DEVICE_NAME.test(safe)) {
    safe = `_${safe}`
  }

  return boundFilename(safe)
}

function sourceFilename(rawUrl?: string): string {
  const value = String(rawUrl || '').trim()

  if (!value || /^data:/i.test(value)) {
    return ''
  }

  try {
    return safeBasename(new URL(value).pathname)
  } catch {
    return safeBasename(value)
  }
}

function normalizedExtension(extension?: string): string {
  const value = String(extension || '')
    .trim()
    .toLowerCase()

  if (!value) {
    return '.png'
  }

  return value.startsWith('.') ? value : `.${value}`
}

function withExtension(filename: string, extension: string): string {
  const currentExtension = filename.match(/\.[a-z0-9]{2,5}$/i)?.[0] || ''

  if (!currentExtension) {
    return boundFilename(`${filename}${extension}`)
  }

  const canonical = (value: string) => (value.toLowerCase() === '.jpeg' ? '.jpg' : value.toLowerCase())

  const complete =
    canonical(currentExtension) === canonical(extension)
      ? filename
      : `${filename.slice(0, -currentExtension.length)}${extension}`

  return boundFilename(complete)
}

export function imageSaveFilename(
  rawUrl?: string,
  suggestedFilename?: string,
  extension?: string,
  now = new Date(),
  uniqueSuffix = randomBytes(3).toString('hex')
): string {
  const ext = normalizedExtension(extension)
  const suggested = safeBasename(suggestedFilename)

  if (suggested) {
    return withExtension(suggested, ext)
  }

  const source = sourceFilename(rawUrl)

  if (source) {
    return withExtension(source, ext)
  }

  const iso = now.toISOString()
  const stamp = `${iso.slice(0, 10).replace(/-/g, '')}-${iso.slice(11, 19).replace(/:/g, '')}-${iso.slice(20, 23)}`

  return `hermes-image-${stamp}-${uniqueSuffix}${ext}`
}

export function imageFormatExtension(buffer: Buffer): string {
  const pngSignature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

  if (buffer.length >= pngSignature.length && buffer.subarray(0, pngSignature.length).equals(pngSignature)) {
    return '.png'
  }

  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return '.jpg'
  }

  const header = buffer.subarray(0, 12).toString('ascii')

  if (header.startsWith('GIF87a') || header.startsWith('GIF89a')) {
    return '.gif'
  }

  if (header.startsWith('RIFF') && header.slice(8, 12) === 'WEBP') {
    return '.webp'
  }

  if (header.startsWith('BM')) {
    return '.bmp'
  }

  const text = buffer
    .subarray(0, 64 * 1024)
    .toString('utf8')
    .replace(/^\uFEFF/, '')
    .trimStart()

  if (/^(?:<\?xml[^>]*>\s*)?(?:<!--[\s\S]*?-->\s*)*<svg\b/i.test(text)) {
    return '.svg'
  }

  return ''
}
