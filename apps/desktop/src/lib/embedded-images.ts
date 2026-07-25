const DATA_URL_RE = /^data:([\w./+-]+);base64,(.*)$/i
const DATA_IMAGE_PREFIX = 'data:image/'
const BASE64_MARKER = ';base64,'
const MIN_EMBEDDED_IMAGE_BASE64_LENGTH = 64
const JSON_IMAGE_OPEN_RE = /\{\s*"type"\s*:\s*"image_url"\s*,\s*"image_url"\s*:\s*\{\s*"url"\s*:\s*"$/
const JSON_IMAGE_CLOSE_RE = /^"\s*\}\s*\}/
const JSON_IMAGE_OPEN_MAX = 96
const JSON_IMAGE_CLOSE_MAX = 16

export const DATA_IMAGE_URL_RE = /^data:image\/[\w.+-]+;base64,/i

export interface EmbeddedImageExtraction {
  cleanedText: string
  images: string[]
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

function isImageMimeCode(code: number): boolean {
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

function isBase64Code(code: number): boolean {
  return (
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    code === 43 ||
    code === 47 ||
    code === 61
  )
}

function readDataImageUrl(text: string, start: number): { end: number; url: string } | null {
  if (!text.startsWith(DATA_IMAGE_PREFIX, start)) {
    return null
  }

  let cursor = start + DATA_IMAGE_PREFIX.length

  while (cursor < text.length && isImageMimeCode(text.charCodeAt(cursor))) {
    cursor += 1
  }

  if (cursor === start + DATA_IMAGE_PREFIX.length || !text.startsWith(BASE64_MARKER, cursor)) {
    return null
  }

  cursor += BASE64_MARKER.length
  const base64Start = cursor

  while (cursor < text.length && isBase64Code(text.charCodeAt(cursor))) {
    cursor += 1
  }

  if (cursor - base64Start < MIN_EMBEDDED_IMAGE_BASE64_LENGTH) {
    return null
  }

  return { end: cursor, url: text.slice(start, cursor) }
}

function embeddedImageRemovalRange(text: string, dataStart: number, dataEnd: number): { end: number; start: number } {
  let start = dataStart
  let end = dataEnd
  const openSearchStart = Math.max(0, dataStart - JSON_IMAGE_OPEN_MAX)
  const openMatch = text.slice(openSearchStart, dataStart).match(JSON_IMAGE_OPEN_RE)

  if (openMatch?.index !== undefined) {
    const close = text.slice(dataEnd, dataEnd + JSON_IMAGE_CLOSE_MAX).match(JSON_IMAGE_CLOSE_RE)

    if (close) {
      start = openSearchStart + openMatch.index
      end = dataEnd + close[0].length
    }
  }

  return { end, start }
}

function normalizeCleanedText(text: string): string {
  return text
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function extractEmbeddedImages(text: string): EmbeddedImageExtraction {
  if (!text || !text.includes(DATA_IMAGE_PREFIX)) {
    return { cleanedText: text, images: [] }
  }

  const images: string[] = []
  const pieces: string[] = []
  let appendCursor = 0
  let searchCursor = 0

  while (searchCursor < text.length) {
    const dataStart = text.indexOf(DATA_IMAGE_PREFIX, searchCursor)

    if (dataStart === -1) {
      break
    }

    const dataUrl = readDataImageUrl(text, dataStart)

    if (!dataUrl) {
      searchCursor = dataStart + DATA_IMAGE_PREFIX.length

      continue
    }

    const range = embeddedImageRemovalRange(text, dataStart, dataUrl.end)
    pieces.push(text.slice(appendCursor, range.start))
    images.push(dataUrl.url)
    appendCursor = range.end
    searchCursor = range.end
  }

  if (!images.length) {
    return { cleanedText: text, images: [] }
  }

  pieces.push(text.slice(appendCursor))

  return { cleanedText: normalizeCleanedText(pieces.join('')), images }
}

export function embeddedImageUrls(text: string): string[] {
  return extractEmbeddedImages(text).images
}

export function textWithoutEmbeddedImages(text: string): string {
  return extractEmbeddedImages(text).cleanedText
}

// The gateway persists attached images as `@image:<path>` directive lines
// (see tui_gateway/server.py's persist-time rewrite), prepended before the
// user's own text. The composer's own optimistic/local turn never carries
// this prefix — it keeps the attachment as separate `attachmentRefs`
// metadata, not inline text. Comparing raw chatMessageText between the
// optimistic turn and the authoritative (persisted) turn therefore always
// mismatches whenever an image was attached, which defeats the "is this the
// same turn" checks in preserveLocalPendingTurnMessages / appendLiveSessionProjection
// and re-appends the optimistic row as if it were a distinct, unconfirmed
// turn — a duplicated user bubble. Strip the directive line(s) before any
// such equality comparison so both sides reduce to the same visible text.
const IMAGE_REF_LINE_RE = /^@image:[^\n]*\n?/gm

export function textWithoutImageRefs(text: string): string {
  return text.replace(IMAGE_REF_LINE_RE, '').trim()
}

// Same directive lines as textWithoutImageRefs, but keeps them instead of
// discarding — used when converting persisted server messages into
// ChatMessage/ThreadMessageLike shape, where `@image:<path>` refs need to
// move from inline text into the `attachmentRefs` metadata field (mirroring
// how the local optimistic composer represents attachments) rather than stay
// embedded in the bubble's clamped text body, where a large inline thumbnail
// pushes the caption text out of the clamp's visible area.
export function extractImageRefs(text: string): { cleanedText: string; refs: string[] } {
  const refs: string[] = []
  const cleanedText = text
    .replace(IMAGE_REF_LINE_RE, match => {
      refs.push(match.trim())

      return ''
    })
    .trim()

  return { cleanedText, refs }
}
