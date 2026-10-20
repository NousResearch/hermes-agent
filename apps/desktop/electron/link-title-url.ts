/**
 * URL admission for the link-title pipeline.
 *
 * The renderer's title-fetch path can send ANY href-shaped string to the
 * ``hermes:fetchLinkTitle`` IPC; the main process must re-validate before
 * anything reaches curl or the hidden title window's ``loadURL()``. A leaked
 * directive-shaped string (``@url:`https://…```, #93893) navigates Chromium to
 * a non-URL and surfaces as repeating ``ERR_NAME_NOT_RESOLVED`` console noise.
 *
 * Pure and dependency-free so the admission rule is testable without
 * Electron; ``main.ts`` imports both helpers.
 */

/** True only for strings that parse as absolute http(s) URLs. */
export function isFetchableHttpUrl(raw: string): boolean {
  let url: URL

  try {
    url = new URL(raw)
  } catch {
    return false
  }

  return url.protocol === 'http:' || url.protocol === 'https:'
}

/**
 * Cache key for a fetched title: host + normalized path + search, or '' when
 * the input is not a parseable URL (never the raw string — an unparseable
 * value must not become a loadable-looking key).
 */
export function canonicalTitleCacheKey(rawUrl: string): string {
  const value = String(rawUrl || '').trim()

  if (!value) {
    return ''
  }

  try {
    const url = new URL(value)
    const host = url.hostname.replace(/^www\./i, '').toLowerCase()
    const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

    return `${host}${pathname}${url.search || ''}`
  } catch {
    // Not a parseable URL (e.g. leaked @url: markup): an empty key makes every
    // consumer bail out instead of feeding the string downstream (#93893).
    return ''
  }
}
