import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'

export const MAX_PASTED_TEXT_CHARS = 120_000
export const MAX_INLINE_DATA_IMAGE_CHARS = 2_000_000
export const MAX_CLIPBOARD_IMAGE_BYTES = 12 * 1024 * 1024

export function pasteLimitMessage(kind: 'image' | 'text'): string {
  if (kind === 'image') {
    return 'That pasted image is too large for inline rendering. Save it as a file and attach it instead.'
  }

  return 'That paste is too large for the Desktop composer. Save it as a .txt/.md file and attach or reference it instead.'
}

export function isOversizedPastedText(text: string): boolean {
  return text.length > MAX_PASTED_TEXT_CHARS
}

export function isOversizedDataImageUrl(text: string): boolean {
  const trimmed = text.trim()

  return DATA_IMAGE_URL_RE.test(trimmed) && trimmed.length > MAX_INLINE_DATA_IMAGE_CHARS
}

export interface TriggerState {
  kind: '@' | '/'
  query: string
  tokenLength: number
}

// `@` triggers stop at the first whitespace — `@file:path` and `@diff` are
// single tokens. `/` triggers keep going so the popover stays live while the
// user types args (`/personality alic` → arg completer suggests `alice`).
// Restricting the slash command name to `[a-zA-Z][\w-]*` avoids matching file
// paths like `src/foo/bar`.
const AT_TRIGGER_RE = /(?:^|[\s])(@)([^\s@/]*)$/
const SLASH_TRIGGER_RE = /(?:^|[\s])(\/)((?:[a-zA-Z][\w-]*(?:\s+\S*)*)?)$/

/** Stable key for paste dedupe — `items` and `files` often mirror the same image as different objects. */
export function blobDedupeKey(blob: Blob): string {
  if (blob instanceof File) {
    return `file:${blob.name}:${blob.size}:${blob.type}:${blob.lastModified}`
  }

  return `blob:${blob.size}:${blob.type}`
}

export function extractClipboardImageBlobs(clipboard: DataTransfer): Blob[] {
  const blobs: Blob[] = []
  const seen = new Set<string>()

  const push = (blob: Blob | null) => {
    if (!blob || blob.size === 0 || blob.size > MAX_CLIPBOARD_IMAGE_BYTES) {
      return
    }

    const key = blobDedupeKey(blob)

    if (seen.has(key)) {
      return
    }

    seen.add(key)
    blobs.push(blob)
  }

  if (clipboard.items?.length) {
    for (const item of clipboard.items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        push(item.getAsFile())
      }
    }
  }

  // Chromium/Electron expose the same pasted image on both `items` and `files`.
  if (blobs.length === 0 && clipboard.files?.length) {
    for (let i = 0; i < clipboard.files.length; i += 1) {
      const file = clipboard.files.item(i)

      if (file && file.type.startsWith('image/')) {
        push(file)
      }
    }
  }

  if (blobs.length > 0) {
    return blobs
  }

  const text = clipboard.getData('text/plain').trim()

  if (DATA_IMAGE_URL_RE.test(text) && !isOversizedDataImageUrl(text)) {
    push(dataUrlToBlob(text))
  }

  if (blobs.length === 0) {
    const html = clipboard.getData('text/html')

    if (html) {
      const matches = html.matchAll(/<img\b[^>]*?\bsrc\s*=\s*["'](data:image\/[^"']+)["']/gi)

      for (const match of matches) {
        if (!isOversizedDataImageUrl(match[1])) {
          push(dataUrlToBlob(match[1]))
        }
      }
    }
  }

  return blobs
}

/** Caret-anchored text before the cursor, or null if the selection isn't a collapsed caret inside `editor`. */
export function textBeforeCaret(editor: HTMLDivElement): string | null {
  const sel = window.getSelection()
  const range = sel?.rangeCount ? sel.getRangeAt(0) : null

  if (!range?.collapsed || !editor.contains(range.commonAncestorContainer)) {
    return null
  }

  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.startContainer, range.startOffset)

  return before.toString()
}

export function detectTrigger(textBefore: string): TriggerState | null {
  const slash = SLASH_TRIGGER_RE.exec(textBefore)

  if (slash) {
    return { kind: '/', query: slash[2], tokenLength: 1 + slash[2].length }
  }

  const at = AT_TRIGGER_RE.exec(textBefore)

  if (at) {
    return { kind: '@', query: at[2], tokenLength: 1 + at[2].length }
  }

  return null
}
