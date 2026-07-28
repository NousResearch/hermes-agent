/**
 * Bare-path recognition for the composer. Walking into a folder from the `@`
 * popover (Tab) re-types the token as a bare `@apps/desktop/` so the next
 * `complete.path` lists that folder's children. That's right while typing, but
 * a bare token is not a reference: `REFERENCE_PATTERN` only matches
 * `@kind:value`, so a draft sent with one attaches nothing and renders as
 * plain text instead of a chip.
 *
 * So the bare token is promoted to its typed form on the way out — the same
 * shape `url-refs.ts` uses for bare links.
 */
import type { KeyboardEvent } from 'react'

import { quoteRefValue, REF_RE, refChipElement, replaceBeforeCaret } from './rich-editor'
import { textBeforeCaret } from './text-utils'

// A `/` is required, exactly like URL_RE requires an explicit scheme: `@teknium1`
// and `@diff` are a handle and a simple ref, not paths, and guessing wrong turns
// someone's name into a file reference. A separator is the cheap signal that the
// user meant a path. The token also can't start with a `:` kind prefix — that is
// already a typed ref and REF_RE owns it.
const BARE_PATH_RE = /(?<![\w@/])@((?!(?:file|folder|url|image|tool|line|terminal|session|git):)[^\s@:]*\/[^\s@:]*)/g
const TYPED_BARE_PATH_RE = new RegExp(`${BARE_PATH_RE.source}$`)

/** Trailing `/` means a directory — that's how the gateway's completion emits
 *  folders, and what Tab-descending leaves behind. */
export function barePathRef(path: string) {
  const trimmed = path.replace(/\/+$/, '')

  return trimmed ? `@${path.endsWith('/') ? 'folder' : 'file'}:${quoteRefValue(trimmed)}` : null
}

/** Rewrite bare `@path` tokens as typed `@file:`/`@folder:` directives, leaving
 *  anything already inside a directive alone. Returns `text` unchanged when
 *  there are none. */
export function pathifyRefs(text: string) {
  if (!text.includes('@')) {
    return text
  }

  REF_RE.lastIndex = 0

  const fenced = Array.from(text.matchAll(REF_RE)).map(match => {
    const start = match.index ?? 0

    return { end: start + match[0].length, start }
  })

  let out = ''
  let cursor = 0

  for (const match of text.matchAll(BARE_PATH_RE)) {
    const start = match.index ?? 0
    const ref = barePathRef(match[1] || '')

    if (!ref || fenced.some(span => start >= span.start && start < span.end)) {
      continue
    }

    out += `${text.slice(cursor, start)}${ref}`
    cursor = start + match[0].length
  }

  return out + text.slice(cursor)
}

/** A plain space finishing a typed `@path` commits it as a chip, so a hand-typed
 *  path chips the same way a picked one does. Returns whether it ran, so a
 *  keydown handler can fall through on anything else. */
export function chipTypedPathOnSpace(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== ' ' || event.metaKey || event.ctrlKey || event.altKey) {
    return false
  }

  const editor = event.currentTarget

  // Runs on every space, so bail on the cheap native read before paying for the
  // caret range walk (same guard shape as the trigger detector).
  if (!editor.textContent?.includes('@')) {
    return false
  }

  const before = textBeforeCaret(editor)
  const match = before ? TYPED_BARE_PATH_RE.exec(before) : null
  const token = match?.[0]
  const ref = match?.[1] ? barePathRef(match[1]) : null

  if (!token || !ref) {
    return false
  }

  const directive = ref.match(/^@([^:]+):(.+)$/)

  if (!directive) {
    return false
  }

  const fragment = document.createDocumentFragment()

  fragment.append(refChipElement(directive[1], directive[2]), document.createTextNode(' '))

  return replaceBeforeCaret(editor, token.length, fragment)
}
