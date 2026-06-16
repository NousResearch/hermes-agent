const EMBEDDED_IMAGE_PREFIX_RE = /\{\s*"type"\s*:\s*"image_url"\s*,\s*"image_url"\s*:\s*\{\s*"url"\s*:\s*"$/
const EMBEDDED_IMAGE_SUFFIX_RE = /^"\s*\}\s*\}/
const DATA_IMAGE_PREFIX_RE = /^data:image\/[\w.+-]+;base64,/i
const BASE64_CHAR_RE = /^[A-Za-z0-9+/=]$/

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

export function extractEmbeddedImages(text: string): EmbeddedImageExtraction {
  if (!text || !text.includes('data:image/')) {
    return { cleanedText: text, images: [] }
  }

  const images: string[] = []
  const cleanedParts: string[] = []
  let cursor = 0
  let searchFrom = 0

  while (searchFrom < text.length) {
    const urlStart = text.indexOf('data:image/', searchFrom)

    if (urlStart < 0) {
      break
    }

    const candidate = text.slice(urlStart, Math.min(text.length, urlStart + 128))
    const prefixMatch = DATA_IMAGE_PREFIX_RE.exec(candidate)

    if (!prefixMatch) {
      searchFrom = urlStart + 'data:image/'.length
      continue
    }

    let urlEnd = urlStart + prefixMatch[0].length

    while (urlEnd < text.length && BASE64_CHAR_RE.test(text[urlEnd])) {
      urlEnd += 1
    }

    // Keep short / malformed data URLs inline. The renderer only lifts real
    // embedded images; tiny strings are usually examples or partial input.
    if (urlEnd - urlStart - prefixMatch[0].length < 64) {
      searchFrom = urlEnd
      continue
    }

    let matchStart = urlStart
    let matchEnd = urlEnd
    const before = text.slice(Math.max(cursor, urlStart - 128), urlStart)
    const prefixEnvelope = EMBEDDED_IMAGE_PREFIX_RE.exec(before)

    if (prefixEnvelope) {
      matchStart = urlStart - prefixEnvelope[0].length
    }

    const suffixEnvelope = EMBEDDED_IMAGE_SUFFIX_RE.exec(text.slice(urlEnd, urlEnd + 32))

    if (suffixEnvelope) {
      matchEnd = urlEnd + suffixEnvelope[0].length
    }

    cleanedParts.push(text.slice(cursor, matchStart))
    images.push(text.slice(urlStart, urlEnd))
    cursor = matchEnd
    searchFrom = matchEnd
  }

  if (images.length === 0) {
    return { cleanedText: text, images: [] }
  }

  cleanedParts.push(text.slice(cursor))

  const cleanedText = cleanedParts
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
