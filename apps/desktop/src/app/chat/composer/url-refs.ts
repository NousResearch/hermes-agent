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

/** A URL needs a valid HTTP(S) authority and host to be worth treating as a URL reference. */
export const hasHttpUrlHost = (url: string) => {
  try {
    const parsed = new URL(splitUrlTail(url).url)

    return (parsed.protocol === 'http:' || parsed.protocol === 'https:') && Boolean(parsed.hostname)
  } catch {
    return false
  }
}

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

    if (!hasHttpUrlHost(url) || fenced.some(span => start >= span.start && start < span.end)) {
      continue
    }

    out += `${text.slice(cursor, start)}@url:${quoteRefValue(url)}`
    cursor = start + url.length
  }

  return out + text.slice(cursor)
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

  if (!hasHttpUrlHost(url)) {
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
