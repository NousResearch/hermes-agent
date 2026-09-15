import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'
import { $reactionsEnabled } from '@/store/reactions-enabled'

import { serializeTextBefore } from './rich-editor'

export interface TriggerState {
  /** True for a `/` typed mid-message — an inline skill/command reference in
   *  prose rather than a command invocation. Arg completion doesn't apply. */
  inline?: boolean
  kind: '@' | '/' | ':'
  query: string
  /** The `@kind:` prefix the user scoped the browse to, when there is one. */
  scope?: DirectiveScope
  tokenLength: number
  /** `query` minus the `scope:` prefix — the value actually being typed. */
  value: string
}

/** Directive kinds the `@` popover can scope a browse to. Mirrors the starter
 *  rows in use-at-completions and the gateway's `complete.path` prefixes. */
export const DIRECTIVE_SCOPES = ['file', 'folder', 'url', 'image', 'tool', 'git'] as const

export type DirectiveScope = (typeof DIRECTIVE_SCOPES)[number]

// Picking "attach a folder" types `@folder:` into the editor, and everything
// after it is the value being browsed. Parsing that prefix off the query is
// what lets the rest of the composer treat it as the BROWSE MODE it is rather
// than characters the user has to maintain by hand — Tab-descending has to
// carry it down, Backspace has to drop it whole, and a chip landing on it has
// to consume it.
const AT_SCOPE_RE = new RegExp(`^(${DIRECTIVE_SCOPES.join('|')}):(.*)$`)

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

const INLINE_IMAGE_SRC_RE = /<img\b[^>]*?\bsrc\s*=\s*["'](data:image\/[^"']+)["']/gi
// Below this, an inline data URL is chrome rather than content — a spacer, a
// 1×1 tracker, or a blurhash placeholder. Real pasted artwork clears it easily.
const MIN_INLINE_IMAGE_BYTES = 4096

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

    return blobs
  }

  // Inline `<img src="data:…">` in the clipboard's HTML — but only for a copy
  // that carried no text of its own. A rich-text copy WITH prose is a text
  // paste that happens to contain images, and its data URLs are the page's
  // decorations rather than content: Discord ships a 32×5 blurhash placeholder
  // beside every image embed, so copying a thread attached a blank thumbnail
  // and (because an image paste swallows the event) dropped the text entirely.
  if (!text) {
    for (const match of clipboard.getData('text/html').matchAll(INLINE_IMAGE_SRC_RE)) {
      const blob = dataUrlToBlob(match[1])

      if (blob && blob.size >= MIN_INLINE_IMAGE_BYTES) {
        push(blob)
      }
    }
  }

  return blobs
}

/** `file:///home/me/my%20report.pdf` → `/home/me/my report.pdf`, or null for
 *  anything that isn't a local-path file URL (remote hosts, non-file schemes). */
export function localPathFromFileUrl(value: string): string | null {
  const raw = value.trim()

  if (!raw.toLowerCase().startsWith('file://')) {
    return null
  }

  const rest = raw.slice('file://'.length)

  if (rest.startsWith('/')) {
    return decodeURIComponent(rest)
  }

  // file://C:/... (single-slash Windows spelling) and file://localhost/...
  // resolve to a path on THIS machine only when the host is empty or loopback.
  const slash = rest.indexOf('/')

  if (slash < 0) {
    return null
  }

  const host = rest.slice(0, slash)

  if (host && host.toLowerCase() !== 'localhost') {
    return null
  }

  return decodeURIComponent(rest.slice(slash))
}

/**
 * Absolute local paths pasted from an OS file-manager copy (Finder "Copy",
 * Explorer "Copy file", Nautilus "Copy"). Chromium surfaces those clipboard
 * entries to the paste event only as `text/uri-list` / `text/plain` file://
 * URLs — never as `items`/`files` entries — so the image-only extraction above
 * never sees them and the paste used to land as literal URL text the gateway
 * can't resolve. Order mirrors the clipboard's own: uri-list first, then
 * plain text. Non-file:// entries are ignored (real URL pastes must stay
 * links), as are entries that decode to nothing.
 */
export function pastedFileClipboardPaths(clipboard: DataTransfer): string[] {
  const paths: string[] = []
  const seen = new Set<string>()

  for (const flavor of ['text/uri-list', 'text/plain']) {
    const raw = clipboard.getData(flavor)

    if (!raw) {
      continue
    }

    for (const line of raw.split(/\r?\n/)) {
      const path = localPathFromFileUrl(line)

      if (path && !seen.has(path)) {
        seen.add(path)
        paths.push(path)
      }
    }
  }

  return paths
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

  return serializeTextBefore(editor, range.startContainer, range.startOffset)
}

/** How many characters of directive scope the caret is sitting inside (`@url:`
 *  with nothing typed after it), or 0. A paste lands INTO that scope: the scope
 *  text is consumed rather than left in front of the chip as leftover syntax. */
export function openDirectiveScope(editor: HTMLDivElement): number {
  const trigger = detectTrigger(textBeforeCaret(editor) ?? '')

  return trigger?.kind === '@' && trigger.scope && !trigger.value ? trigger.tokenLength : 0
}

export function detectTrigger(textBefore: string): TriggerState | null {
  // An inline `/skill` is a reference dropped into prose, so it carries no args
  // and the whole match is the token the chip replaces. Checked before the
  // anchored command shape so a second slash isn't mistaken for the first
  // command's argument.
  const inline = SLASH_INLINE_TRIGGER_RE.exec(textBefore)

  if (inline) {
    const query = inline[2] ?? ''

    return { inline: true, kind: '/', query, tokenLength: 1 + query.length, value: query }
  }

  const command = SLASH_COMMAND_TRIGGER_RE.exec(textBefore)

  if (command) {
    return { kind: '/', query: command[2], tokenLength: 1 + command[2].length, value: command[2] }
  }

  const at = AT_TRIGGER_RE.exec(textBefore)

  if (at) {
    const query = at[2]
    const scoped = AT_SCOPE_RE.exec(query)

    return {
      kind: '@',
      query,
      ...(scoped ? { scope: scoped[1] as DirectiveScope } : {}),
      tokenLength: 1 + query.length,
      value: scoped ? (scoped[2] ?? '') : query
    }
  }

  // After `@` so a directive starter's colon (`@file:`) stays an `@` query.
  // Rides the reactions opt-in (Settings → Appearance) — both are one
  // "emoji features" surface, off by default.
  const emoji = $reactionsEnabled.get() ? EMOJI_TRIGGER_RE.exec(textBefore) : null

  if (emoji) {
    return { kind: ':', query: emoji[2], tokenLength: 1 + emoji[2].length, value: emoji[2] }
  }

  return null
}
