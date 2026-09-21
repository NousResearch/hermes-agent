/**
 * Charset-aware decode for the curl title tier.
 *
 * A non-UTF-8 page (Big5, GBK, Shift-JIS, EUC-KR, …) decoded blindly as UTF-8
 * collapses entirely to U+FFFD replacement characters — `Buffer#toString('utf8')`
 * never looks at the declared charset, and neither does `Response.text()`. This
 * reads the charset the server declared in `Content-Type`, and failing that,
 * sniffs the page's own `<meta charset>` — readable via a plain UTF-8 decode,
 * since a legacy encoding's ASCII-range bytes, which is all a `<meta>` tag ever
 * uses, decode identically under UTF-8.
 *
 * Node's full-ICU build supports these legacy labels directly via `TextDecoder`.
 */

const CONTENT_TYPE_CHARSET_RE = /charset=["']?([\w-]+)/i
const META_CHARSET_RE = /<meta[^>]+charset=["']?([\w-]+)/i

function decode(bytes: Uint8Array, label: string): string {
  try {
    return new TextDecoder(label).decode(bytes)
  } catch {
    return ''
  }
}

export function decodeHttpBody(bytes: Uint8Array, contentTypeHeader: string): string {
  const utf8 = decode(bytes, 'utf-8')
  const declared = (contentTypeHeader.match(CONTENT_TYPE_CHARSET_RE)?.[1] ?? '').toLowerCase()
  const label = declared || (utf8.match(META_CHARSET_RE)?.[1] ?? '').toLowerCase()

  return label && label !== 'utf-8' && label !== 'utf8' ? decode(bytes, label) || utf8 : utf8
}
