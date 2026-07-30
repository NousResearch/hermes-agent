import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'
import { $reactionsEnabled } from '@/store/reactions-enabled'

export interface TriggerState {
  /** True for a `/` typed mid-message — an inline skill/command reference in
   *  prose rather than a command invocation. Arg completion doesn't apply. */
  inline?: boolean
  kind: '@' | '/' | ':'
  query: string
  tokenLength: number
}

// `@` triggers stop at the first whitespace — `@file:path` and `@diff` are
// single tokens, and a path is part of that token: `@./src/`, `@~/Desktop/`,
// and `@file:src/foo` all have to keep the popover live while the user walks
// into subdirectories. Excluding `/` from the query class would end the token
// at the first separator, which is exactly the "can't browse into a folder"
// bug. Restricting the slash command name to `[a-zA-Z][\w-]*` avoids
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
// Only the FIRST slash can be an invocation, so the inline shape is tested
// first: the command regex's argument tail (`(?:\s+\S*)*`) happily swallows a
// later `/skill` as if it were an argument, which killed completion for every
// slash after a leading command (`/work /cle` → nothing).
//
// The inline shape is what makes skills reachable anywhere in a prompt. Both
// shapes need the trailing `$`: detection runs against the text BEFORE the
// caret, so the match must end where the user is typing.
//
// U+FFFC is the placeholder textBeforeCaret emits for a committed chip. A chip
// edge is a token boundary just like whitespace (upstream assistant-ui's
// Lexical DirectivePlugin gets the same semantics from node boundaries), so
// `@` or `/` typed immediately after a pill still opens the popover.
const AT_TRIGGER_RE = /(?:^|[\s\uFFFC])(@)([^\s@\uFFFC]*)$/
const SLASH_COMMAND_TRIGGER_RE = /^(\/)((?:[a-zA-Z][\w-]*(?:\s+\S*)*)?)$/
const SLASH_INLINE_TRIGGER_RE = /[\s\uFFFC](\/)([a-zA-Z][\w-]*)?$/
// `:joy` → emoji completions, Slack-style. Boundary-anchored so a mid-word
// colon (`localhost:8080`, `note:`) never fires; two chars minimum so a bare
// `:` or `:D` smiley doesn't open a popover the user didn't ask for.
const EMOJI_TRIGGER_RE = /(?:^|[\s\uFFFC])(:)([a-zA-Z0-9_+-]{2,})$/

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

/** Caret-anchored text before the cursor, or null if the selection isn't a
 *  collapsed caret inside `editor`.
 *
 *  Chips are ATOMIC to trigger detection: a committed pill must not leak its
 *  label text into the string the trigger regexes see. A `/work` pill whose
 *  label serialized into this text made the `^`-anchored command regex treat
 *  everything after it as that command's argument — which silenced the `@`
 *  popover for the rest of the message (`/work @Desk` → no trigger → the
 *  typed path never chips and submits as plain text). Each chip contributes
 *  an object-replacement placeholder instead, and <br> contributes a newline
 *  so a trigger at the start of a wrapped line still detects. */
export function textBeforeCaret(editor: HTMLDivElement): string | null {
  const sel = window.getSelection()
  const range = sel?.rangeCount ? sel.getRangeAt(0) : null

  if (!range?.collapsed || !editor.contains(range.commonAncestorContainer)) {
    return null
  }

  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.startContainer, range.startOffset)

  const scratch = document.createElement('div')

  scratch.append(before.cloneContents())

  for (const chip of scratch.querySelectorAll('[data-ref-text]')) {
    chip.replaceWith('\uFFFC')
  }

  for (const br of scratch.querySelectorAll('br')) {
    br.replaceWith('\n')
  }

  return scratch.textContent ?? ''
}

export function detectTrigger(textBefore: string): TriggerState | null {
  // An inline `/skill` is a reference dropped into prose, so it carries no args
  // and the whole match is the token the chip replaces. Checked before the
  // anchored command shape so a second slash isn't mistaken for the first
  // command's argument.
  const inline = SLASH_INLINE_TRIGGER_RE.exec(textBefore)

  if (inline) {
    const query = inline[2] ?? ''

    return { inline: true, kind: '/', query, tokenLength: 1 + query.length }
  }

  const command = SLASH_COMMAND_TRIGGER_RE.exec(textBefore)

  if (command) {
    return { kind: '/', query: command[2], tokenLength: 1 + command[2].length }
  }

  const at = AT_TRIGGER_RE.exec(textBefore)

  if (at) {
    return { kind: '@', query: at[2], tokenLength: 1 + at[2].length }
  }

  // After `@` so a directive starter's colon (`@file:`) stays an `@` query.
  // Rides the reactions opt-in (Settings → Appearance) — both are one
  // "emoji features" surface, off by default.
  const emoji = $reactionsEnabled.get() ? EMOJI_TRIGGER_RE.exec(textBefore) : null

  if (emoji) {
    return { kind: ':', query: emoji[2], tokenLength: 1 + emoji[2].length }
  }

  return null
}
