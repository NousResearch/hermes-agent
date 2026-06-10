const DATA_URL_RE = /^data:([\w./+-]+);base64,(.*)$/i

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

const DATA_IMAGE_PREFIX = 'data:image/'
const BASE64_MARKER = ';base64,'
const MIN_BASE64_LENGTH = 64

const MIME_CHAR_RE = /[\w.+-]/
const WHITESPACE_RE = /\s/

// `{"type": "image_url", "image_url": {"url": "<data url>"}}` with optional
// whitespace between tokens; prefix and suffix are stripped independently.
const WRAPPER_PREFIX_TOKENS = ['{', '"type"', ':', '"image_url"', ',', '"image_url"', ':', '{', '"url"', ':', '"']
const WRAPPER_SUFFIX_TOKENS = ['"', '}', '}']

function isBase64Code(code: number): boolean {
  return (
    (code >= 48 && code <= 57) || // 0-9
    (code >= 65 && code <= 90) || // A-Z
    (code >= 97 && code <= 122) || // a-z
    code === 43 || // +
    code === 47 || // /
    code === 61 // =
  )
}

function wrapperPrefixStart(text: string, urlStart: number): number {
  let i = urlStart

  for (let t = WRAPPER_PREFIX_TOKENS.length - 1; t >= 0; t -= 1) {
    const token = WRAPPER_PREFIX_TOKENS[t]
    const tokenStart = i - token.length

    if (tokenStart < 0 || !text.startsWith(token, tokenStart)) {
      return -1
    }

    i = tokenStart

    if (t > 0) {
      while (i > 0 && WHITESPACE_RE.test(text[i - 1])) {
        i -= 1
      }
    }
  }

  return i
}

function wrapperSuffixEnd(text: string, urlEnd: number): number {
  let i = urlEnd

  for (let t = 0; t < WRAPPER_SUFFIX_TOKENS.length; t += 1) {
    if (t > 0) {
      while (i < text.length && WHITESPACE_RE.test(text[i])) {
        i += 1
      }
    }

    if (!text.startsWith(WRAPPER_SUFFIX_TOKENS[t], i)) {
      return -1
    }

    i += WRAPPER_SUFFIX_TOKENS[t].length
  }

  return i
}

interface EmbeddedImageMatch {
  start: number
  end: number
  dataUrl: string
}

// Linear scan replacing the previous whole-text regex, whose backtracking
// overflowed V8's regexp stack on multi-megabyte base64 payloads.
function findEmbeddedImage(text: string, from: number): EmbeddedImageMatch | null {
  let searchFrom = from

  while (searchFrom < text.length) {
    const urlStart = text.indexOf(DATA_IMAGE_PREFIX, searchFrom)

    if (urlStart === -1) {
      return null
    }

    searchFrom = urlStart + 1

    let i = urlStart + DATA_IMAGE_PREFIX.length
    const mimeStart = i

    while (i < text.length && MIME_CHAR_RE.test(text[i])) {
      i += 1
    }

    if (i === mimeStart || !text.startsWith(BASE64_MARKER, i)) {
      continue
    }

    const base64Start = i + BASE64_MARKER.length
    let urlEnd = base64Start

    while (urlEnd < text.length && isBase64Code(text.charCodeAt(urlEnd))) {
      urlEnd += 1
    }

    if (urlEnd - base64Start < MIN_BASE64_LENGTH) {
      continue
    }

    const prefixStart = wrapperPrefixStart(text, urlStart)
    const suffixEnd = wrapperSuffixEnd(text, urlEnd)

    return {
      start: prefixStart === -1 ? urlStart : prefixStart,
      end: suffixEnd === -1 ? urlEnd : suffixEnd,
      dataUrl: text.slice(urlStart, urlEnd),
    }
  }

  return null
}

export function extractEmbeddedImages(text: string): EmbeddedImageExtraction {
  if (!text || !text.includes(DATA_IMAGE_PREFIX)) {
    return { cleanedText: text, images: [] }
  }

  const images: string[] = []
  const kept: string[] = []
  let cursor = 0

  for (let match = findEmbeddedImage(text, 0); match; match = findEmbeddedImage(text, match.end)) {
    images.push(match.dataUrl)
    kept.push(text.slice(cursor, match.start))
    cursor = match.end
  }

  kept.push(text.slice(cursor))

  const cleanedText = kept
    .join('')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

  return { cleanedText, images }
}

export function embeddedImageUrls(text: string): string[] {
  return extractEmbeddedImages(text).images
}

export function textWithoutEmbeddedImages(text: string): string {
  return extractEmbeddedImages(text).cleanedText
}
