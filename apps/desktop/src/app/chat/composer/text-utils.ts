import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'

export interface TriggerState {
  /** True for a `/` typed mid-message — an inline skill/command reference in
   *  prose rather than a command invocation. Arg completion doesn't apply. */
  inline?: boolean
  kind: '@' | '/'
  query: string
  tokenLength: number
}

// `@` triggers stop at the first whitespace — `@file:path` and `@diff` are
// single tokens. Restricting the slash command name to `[a-zA-Z][\w-]*` avoids
// matching file paths like `src/foo/bar`.
//
// `/` triggers fire in two shapes, because a slash means two different things
// depending on where it sits:
//
//  - At position 0 it's a COMMAND invocation the app executes (SLASH_COMMAND_RE
//    is `^`-anchored, and so is the backend's). The popover stays live past the
//    command name so arg completion works (`/personality alic` → `alice`).
//  - After whitespace it's an inline REFERENCE the user is dropping into prose
//    ("clean this up with /clean"). The text submits as an ordinary message, so
//    there are no args to complete — the trigger is a single token that ends at
//    the next space, exactly like `@`.
//
// The inline shape is what makes skills reachable anywhere in a prompt. Both
// shapes need the trailing `$`: detection runs against the text BEFORE the
// caret, so the match must end where the user is typing.
const AT_TRIGGER_RE = /(?:^|[\s])(@)([^\s@/]*)$/
const SLASH_COMMAND_TRIGGER_RE = /^(\/)((?:[a-zA-Z][\w-]*(?:\s+\S*)*)?)$/
const SLASH_INLINE_TRIGGER_RE = /[\s](\/)([a-zA-Z][\w-]*)?$/

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
  const command = SLASH_COMMAND_TRIGGER_RE.exec(textBefore)

  if (command) {
    return { kind: '/', query: command[2], tokenLength: 1 + command[2].length }
  }

  // An inline `/skill` is a reference dropped into prose, so it carries no args
  // and the whole match is the token the chip replaces.
  const inline = SLASH_INLINE_TRIGGER_RE.exec(textBefore)

  if (inline) {
    const query = inline[2] ?? ''

    return { inline: true, kind: '/', query, tokenLength: 1 + query.length }
  }

  const at = AT_TRIGGER_RE.exec(textBefore)

  if (at) {
    return { kind: '@', query: at[2], tokenLength: 1 + at[2].length }
  }

  return null
}
