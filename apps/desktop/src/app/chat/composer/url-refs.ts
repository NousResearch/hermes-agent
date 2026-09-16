/**
 * Bare-link recognition for the composer. A link the user pastes or types is the
 * same thing the "+ → Add URL" dialog inserts, so it becomes an `@url:`
 * directive: a chip that truncates instead of a wall of URL text, and a
 * reference the gateway resolves.
 */
import type { KeyboardEvent } from 'react'

import { quoteRefValue, REF_RE, refChipElement, replaceBeforeCaret } from './rich-editor'
import { textBeforeCaret } from './text-utils'

// An explicit scheme only — `example.com` bare is too easy to hit by accident
// (a filename, a version, a sentence). Brackets and quotes fence a URL in prose;
// parens don't, so they stay in and an unbalanced tail is trimmed below.
const URL_RE = /https?:\/\/[^\s<>[\]{}"'`]+/gi
const TYPED_URL_RE = /(?:^|\s)(https?:\/\/[^\s<>[\]{}"'`]+)$/i

/** A URL at the end of a sentence carries the punctuation that ended it. */
function splitUrlTail(raw: string) {
  let url = raw.replace(/[,.;:!?]+$/, '')

  while (url.endsWith(')') && url.split(')').length > url.split('(').length) {
    url = url.slice(0, -1)
  }

  return { trailing: raw.slice(url.length), url }
}

/** A URL needs a host past the scheme to be worth chipping. */
const hasHost = (url: string) => /^https?:\/\/[^/\s]/i.test(url)

/** Rewrite bare links in `text` as `@url:` directives, leaving links that are
 *  already part of a directive alone. Returns `text` unchanged when there are
 *  none. */
export function linkifyUrls(text: string) {
  REF_RE.lastIndex = 0

  const fenced = Array.from(text.matchAll(REF_RE)).map(match => {
    const start = match.index ?? 0

    return { end: start + match[0].length, start }
  })

  let out = ''
  let cursor = 0

  for (const match of text.matchAll(URL_RE)) {
    const start = match.index ?? 0
    const { url } = splitUrlTail(match[0])

    if (!hasHost(url) || fenced.some(span => start >= span.start && start < span.end)) {
      continue
    }

    out += `${text.slice(cursor, start)}@url:${quoteRefValue(url)}`
    cursor = start + url.length
  }

  return out + text.slice(cursor)
}

/** The href to apply when a clipboard payload is exactly ONE supported link —
 *  a bare or `<…>`-wrapped `http(s)` URL with a host and nothing else. Null for
 *  anything that isn't a lone link (prose, multiple links, trailing text), so
 *  callers fall through to the normal paste pipeline. Ported from
 *  block/buzz#6684's `resolveExactLinkPaste`. */
export function resolveExactLinkPaste(raw: string): string | null {
  const text = raw.trim()
  const unwrapped = text.startsWith('<') && text.endsWith('>') && text.length > 2 ? text.slice(1, -1).trim() : text

  URL_RE.lastIndex = 0

  const match = URL_RE.exec(unwrapped)

  if (!match || match.index !== 0 || match[0].length !== unwrapped.length) {
    return null
  }

  const { trailing, url } = splitUrlTail(unwrapped)

  // Trailing sentence punctuation means the user copied prose, not a link.
  if (trailing || !hasHost(url)) {
    return null
  }

  return url
}

/** The selected composer text a link paste should hyperlink, or null when the
 *  selection can't take a link mark: collapsed, outside `editor`, spanning
 *  chips or line breaks, or whitespace-only. */
export function selectionLinkLabel(editor: HTMLElement): string | null {
  const selection = window.getSelection()

  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return null
  }

  const range = selection.getRangeAt(0)

  if (!editor.contains(range.commonAncestorContainer)) {
    return null
  }

  const probe = document.createElement('div')

  probe.append(range.cloneContents())

  // A chip inside the selection is a directive, not prose — linking over it
  // would destroy the reference. Multi-line selections don't read as a label.
  if (probe.querySelector('[data-ref-text], br')) {
    return null
  }

  const label = probe.textContent?.replace(/\s+/g, ' ').trim() ?? ''

  return label || null
}

/** Markdown link for a paste-over-selection: the label the user selected, the
 *  URL they pasted. Square brackets in the label are escaped so the link
 *  survives markdown parsing downstream. */
export function markdownLinkFor(label: string, url: string): string {
  return `[${label.replace(/([[\]])/g, '\\$1')}](${url})`
}

/** A plain space finishing a typed link commits it as a chip (followed by
 *  whatever punctuation ended it, then the space). Returns whether it ran, so a
 *  keydown handler can fall through on anything else. */
export function chipTypedUrlOnSpace(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== ' ' || event.metaKey || event.ctrlKey || event.altKey) {
    return false
  }

  const editor = event.currentTarget

  // Runs on every space, so bail on the cheap native read before paying for the
  // caret range walk (same guard shape as the trigger detector).
  if (!editor.textContent?.includes('://')) {
    return false
  }

  const before = textBeforeCaret(editor)
  const match = before ? TYPED_URL_RE.exec(before) : null
  const token = match?.[1]

  if (!token) {
    return false
  }

  const { trailing, url } = splitUrlTail(token)

  if (!hasHost(url)) {
    return false
  }

  const fragment = document.createDocumentFragment()

  fragment.append(refChipElement('url', quoteRefValue(url)))

  if (trailing) {
    fragment.append(document.createTextNode(trailing))
  }

  fragment.append(document.createTextNode(' '))

  return replaceBeforeCaret(editor, token.length, fragment)
}
