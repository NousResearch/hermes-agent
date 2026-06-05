import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'

export interface TriggerState {
  kind: '@' | '/'
  query: string
  tokenLength: number
}

const TRIGGER_RE = /(?:^|[\s])([@/])([^\s@/]*)$/

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
    if (!blob || blob.size === 0) {
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

  if (DATA_IMAGE_URL_RE.test(text)) {
    push(dataUrlToBlob(text))
  }

  if (blobs.length === 0) {
    const html = clipboard.getData('text/html')

    if (html) {
      const matches = html.matchAll(/<img\b[^>]*?\bsrc\s*=\s*["'](data:image\/[^"']+)["']/gi)

      for (const match of matches) {
        push(dataUrlToBlob(match[1]))
      }
    }
  }

  return blobs
}

/**
 * Extract non-image files from a paste clipboard, resolving each File to
 * its filesystem path via `window.hermesDesktop.getPathForFile`.
 *
 * Must be called synchronously from inside the paste handler — `DataTransfer`
 * items are detached as soon as the handler returns, and `webUtils.getPathForFile`
 * also requires the original (non-cloned) File reference.
 */
export function extractClipboardFiles(clipboard: DataTransfer): { file?: File; path: string }[] {
  const result: { file?: File; path: string }[] = []
  const seenPaths = new Set<string>()
  const seenFiles = new Set<File>()
  const getPath = window.hermesDesktop?.getPathForFile

  if (clipboard.files?.length) {
    for (let i = 0; i < clipboard.files.length; i += 1) {
      const file = clipboard.files.item(i)

      if (!file || seenFiles.has(file)) {
        continue
      }

      // Skip images — those are handled separately by extractClipboardImageBlobs.
      if (file.type.startsWith('image/')) {
        continue
      }

      seenFiles.add(file)
      let path = ''

      if (getPath) {
        try {
          path = getPath(file) || ''
        } catch {
          path = ''
        }
      }

      if (path && seenPaths.has(path)) {
        continue
      }

      if (path) {
        seenPaths.add(path)
      }

      result.push({ file, path })
    }
  }

  if (clipboard.items?.length) {
    for (const item of clipboard.items) {
      if (item.kind !== 'file') {
        continue
      }

      const file = item.getAsFile()

      if (!file || seenFiles.has(file)) {
        continue
      }

      if (file.type.startsWith('image/')) {
        continue
      }

      seenFiles.add(file)
      let path = ''

      if (getPath) {
        try {
          path = getPath(file) || ''
        } catch {
          path = ''
        }
      }

      if (path && seenPaths.has(path)) {
        continue
      }

      if (path) {
        seenPaths.add(path)
      }

      result.push({ file, path })
    }
  }

  return result
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
  const match = TRIGGER_RE.exec(textBefore)

  if (!match) {
    return null
  }

  return { kind: match[1] as '@' | '/', query: match[2], tokenLength: 1 + match[2].length }
}
