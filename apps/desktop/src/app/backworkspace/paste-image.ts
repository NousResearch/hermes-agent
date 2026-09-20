import { extractClipboardImageBlobs } from '@/app/chat/composer/text-utils'

// What the clipboard calls a picture and the page can keep. Mirrors the
// suffixes the backend stores (tui_gateway/backworkspace.py): SVG is a document
// that can carry script, so it is left to paste as text like any other file the
// page cannot show.
const IMAGE_SUFFIX_BY_TYPE: Record<string, string> = {
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp'
}

/**
 * The pasted pictures the page can store, in clipboard order.
 *
 * The composer's extractor does the reading: one image reaches the clipboard as
 * `items`, as `files`, as a `data:` URL or inside the HTML, often as several of
 * those at once, and it has already learnt which of them are the picture and
 * which are a rich copy's decorations.
 */
export function storableImages(clipboard: DataTransfer | null): Blob[] {
  return clipboard ? extractClipboardImageBlobs(clipboard).filter(blob => IMAGE_SUFFIX_BY_TYPE[blob.type]) : []
}

/** A file name for the store to read a suffix from — a pasted picture has none. */
export function attachmentName(blob: Blob): string {
  return `clipboard${IMAGE_SUFFIX_BY_TYPE[blob.type] ?? ''}`
}

/** Base64 for the wire, in chunks: `String.fromCharCode(...bytes)` blows the
 *  argument limit on anything bigger than a small screenshot. */
export function base64FromBytes(bytes: Uint8Array): string {
  const CHUNK = 0x8000
  let binary = ''

  for (let at = 0; at < bytes.length; at += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(at, at + CHUNK))
  }

  return btoa(binary)
}

/** How the page links an image it stores: a relative markdown link. */
export function imageMarkdown(href: string): string {
  return `![](${href})`
}

/** The page with `link` on a line of its own at the end — where a picture goes
 *  when the window was turned back before it finished storing. */
export function appendLink(content: string, link: string): string {
  return content && !content.endsWith('\n') ? `${content}\n${link}` : content + link
}
