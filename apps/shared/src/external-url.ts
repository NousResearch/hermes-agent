/**
 * Which URLs Desktop hands to a browser when asked to open something
 * externally. Electron's `openExternalUrl` (shell.openExternal) and the
 * browser-hosted bridge (window.open) share this one rule, so an agent- or
 * tool-supplied `javascript:` or custom-scheme link is refused on both hosts.
 *
 * `file:` is its own kind: only a host with a local file opener may act on it.
 */

const WEB_PROTOCOLS = new Set(['http:', 'https:', 'mailto:'])

// `tel:` opens the OS dialer, so only a plain number qualifies: an optional `+` and 7-15
// digits. Anything else (parameters, letters, separators) could smuggle input to the
// handler app and is refused like any unsupported scheme.
const TELEPHONE_URL_RE = /^tel:\+?\d{7,15}$/iu

export interface ExternalUrlTarget {
  kind: 'file' | 'web'
  /** `web`: the normalized URL. `file`: the trimmed input, left for the host's file opener to parse. */
  url: string
}

/** Classify an external URL; null for empty, malformed or unsupported-scheme input. */
export function externalUrlTarget(rawUrl: string): ExternalUrlTarget | null {
  const raw = String(rawUrl || '').trim()

  if (!raw) {
    return null
  }

  let parsed: URL

  try {
    parsed = new URL(raw)
  } catch {
    return null
  }

  if (parsed.protocol === 'file:') {
    return { kind: 'file', url: raw }
  }

  if (parsed.protocol === 'tel:') {
    return TELEPHONE_URL_RE.test(parsed.toString()) ? { kind: 'web', url: parsed.toString() } : null
  }

  return WEB_PROTOCOLS.has(parsed.protocol) ? { kind: 'web', url: parsed.toString() } : null
}
