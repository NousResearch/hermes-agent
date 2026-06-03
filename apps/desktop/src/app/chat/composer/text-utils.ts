import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'

export interface TriggerState {
  kind: '@' | '/'
  query: string
  tokenLength: number
}

const TRIGGER_RE = /(?:^|[\s])([@/])([^\s@/]*)$/

export function extractClipboardImageBlobs(clipboard: DataTransfer): Blob[] {
  const blobs: Blob[] = []
  const seen = new Set<Blob>()

  const push = (blob: Blob | null) => {
    if (!blob || blob.size === 0 || seen.has(blob)) {
      return
    }

    seen.add(blob)
    blobs.push(blob)
  }

  if (clipboard.items?.length) {
    for (const item of clipboard.items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        push(item.getAsFile())
      }
    }
  }

  if (clipboard.files?.length) {
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
