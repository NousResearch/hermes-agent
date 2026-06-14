const DATA_URL_RE = /^data:([\w./+-]+);base64,(.*)$/i

export const DATA_IMAGE_URL_RE = /^data:image\/[\w.+-]+;base64,/i

export interface EmbeddedImageExtraction {
  cleanedText: string
  images: string[]
}

const DEFAULT_MAX_EMBEDDED_IMAGES = 24
const DATA_IMAGE_PREFIX = 'data:image/'
const BASE64_MARKER = ';base64,'
const MIN_EMBEDDED_IMAGE_BASE64_CHARS = 64
const JSON_IMAGE_URL_OPEN_RE =
  /^\{\s*"type"\s*:\s*"image_url"\s*,\s*"image_url"\s*:\s*\{\s*"url"\s*:\s*"$/i
const JSON_IMAGE_URL_OPEN_LOOKBEHIND = 2_048

function isMimeSubtypeChar(code: number): boolean {
  return (
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    code === 43 ||
    code === 45 ||
    code === 46 ||
    code === 95
  )
}

function isBase64Char(code: number): boolean {
  return (
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    code === 43 ||
    code === 47 ||
    code === 61
  )
}

function isJsonWhitespace(code: number): boolean {
  return code === 9 || code === 10 || code === 13 || code === 32
}

function startsWithIgnoreCase(text: string, value: string, index: number): boolean {
  return text.slice(index, index + value.length).toLowerCase() === value
}

function parseDataImageUrlAt(text: string, start: number): { end: number } | null {
  let index = start + DATA_IMAGE_PREFIX.length
  const subtypeStart = index

  while (index < text.length && isMimeSubtypeChar(text.charCodeAt(index))) {
    index += 1
  }

  if (index === subtypeStart || !startsWithIgnoreCase(text, BASE64_MARKER, index)) {
    return null
  }

  index += BASE64_MARKER.length
  const base64Start = index

  while (index < text.length && isBase64Char(text.charCodeAt(index))) {
    index += 1
  }

  return index - base64Start >= MIN_EMBEDDED_IMAGE_BASE64_CHARS ? { end: index } : null
}

function jsonImageEnvelopeStart(text: string, cursor: number, dataUrlStart: number): number {
  const lookbehindStart = Math.max(cursor, dataUrlStart - JSON_IMAGE_URL_OPEN_LOOKBEHIND)
  let candidateStart = text.indexOf('{', lookbehindStart)

  while (candidateStart !== -1 && candidateStart < dataUrlStart) {
    if (JSON_IMAGE_URL_OPEN_RE.test(text.slice(candidateStart, dataUrlStart))) {
      return candidateStart
    }

    candidateStart = text.indexOf('{', candidateStart + 1)
  }

  return dataUrlStart
}

function consumeJsonImageEnvelopeClose(text: string, start: number): number {
  let index = start

  if (text.charCodeAt(index) !== 34) {
    return start
  }

  index += 1

  while (index < text.length && isJsonWhitespace(text.charCodeAt(index))) {
    index += 1
  }

  if (text.charCodeAt(index) !== 125) {
    return start
  }

  index += 1

  while (index < text.length && isJsonWhitespace(text.charCodeAt(index))) {
    index += 1
  }

  return text.charCodeAt(index) === 125 ? index + 1 : start
}

function normalizeCleanedText(text: string): string {
  return text.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
}

export function dataUrlToBlob(dataUrl: string): Blob | null {
  const match = DATA_URL_RE.exec(dataUrl.trim())

  if (!match) {
    return null
  }

  try {
    const bytes = atob(match[2])
    const buffer = new Uint8Array(bytes.length)

    for (let i = 0; i < bytes.length; i += 1) {
      buffer[i] = bytes.charCodeAt(i)
    }

    return new Blob([buffer], { type: match[1] })
  } catch {
    return null
  }
}

export function extractEmbeddedImages(
  text: string,
  { maxImages = DEFAULT_MAX_EMBEDDED_IMAGES }: { maxImages?: number } = {}
): EmbeddedImageExtraction {
  if (!text || !text.includes('data:image/')) {
    return { cleanedText: text, images: [] }
  }

  const images: string[] = []
  const cleanedChunks: string[] = []
  let cursor = 0
  let searchFrom = 0

  // Avoid running a global regex over multi-megabyte base64 strings; V8 can
  // overflow the renderer stack before React gets a chance to paint an error.
  while (searchFrom < text.length) {
    const dataUrlStart = text.indexOf(DATA_IMAGE_PREFIX, searchFrom)

    if (dataUrlStart === -1) {
      break
    }

    const parsed = parseDataImageUrlAt(text, dataUrlStart)

    if (!parsed) {
      searchFrom = dataUrlStart + DATA_IMAGE_PREFIX.length
      continue
    }

    const matchStart = jsonImageEnvelopeStart(text, cursor, dataUrlStart)
    const matchEnd =
      matchStart === dataUrlStart ? parsed.end : consumeJsonImageEnvelopeClose(text, parsed.end)

    cleanedChunks.push(text.slice(cursor, matchStart))

    if (images.length < maxImages) {
      images.push(text.slice(dataUrlStart, parsed.end))
    }

    cursor = matchEnd
    searchFrom = matchEnd
  }

  cleanedChunks.push(text.slice(cursor))

  const cleanedText = normalizeCleanedText(cleanedChunks.join(''))

  return { cleanedText, images }
}

export function embeddedImageUrls(text: string): string[] {
  return extractEmbeddedImages(text).images
}

export function textWithoutEmbeddedImages(text: string): string {
  return extractEmbeddedImages(text).cleanedText
}
