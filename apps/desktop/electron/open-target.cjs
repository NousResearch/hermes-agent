/**
 * Pure classification for "what does this href mean when the user clicks /
 * opens it" — the decision half of main.cjs's openExternalUrl, split out so it
 * can be unit-tested without an Electron runtime (shell.openPath / openExternal
 * stay in main.cjs as the side-effecting half).
 *
 * Returns one of:
 *   { kind: 'file',  path }   — a local file to hand to shell.openPath. `path`
 *                               is a file:// URL string OR a raw scheme-less
 *                               path (relative / ~), both of which
 *                               resolveRequestedPathForIpc accepts.
 *   { kind: 'web',   url }    — an http/https/mailto URL for shell.openExternal.
 *   { kind: 'reject' }        — empty, or a URL with an unknown scheme.
 *
 * Why scheme-less paths matter: the renderer cannot always build a file:// URL
 * for a media link. A relative path (`out/a.png`) or a `~/...` path has no
 * cwd/home in the renderer, and `file://<relative>` mis-parses the first
 * segment as a URL host (rejected on macOS/Linux). Those arrive here raw and
 * are classified as files so the main process can resolve them against a real
 * base. See lib/media.ts::toLocalFileUrl for the renderer side.
 */

const WEB_PROTOCOLS = new Set(['http:', 'https:', 'mailto:'])

// A scheme-less, non-URL string that looks like a local path (relative or ~).
// We deliberately reject strings that DO carry a `scheme://` so a genuinely
// broken or unknown-scheme link can't masquerade as a file path.
const HAS_URL_SCHEME_RE = /^[a-z][a-z0-9+.-]*:\/\//i

function classifyOpenTarget(rawUrl) {
  const raw = String(rawUrl == null ? '' : rawUrl).trim()
  if (!raw) {
    return { kind: 'reject' }
  }

  let parsed
  try {
    parsed = new URL(raw)
  } catch {
    // Not a parseable URL. Treat a scheme-less string as a local path; reject
    // anything that still looks like `scheme://...` (unknown/broken scheme).
    if (HAS_URL_SCHEME_RE.test(raw)) {
      return { kind: 'reject' }
    }

    return { kind: 'file', path: raw }
  }

  if (parsed.protocol === 'file:') {
    return { kind: 'file', path: parsed.toString() }
  }

  if (WEB_PROTOCOLS.has(parsed.protocol)) {
    return { kind: 'web', url: parsed.toString() }
  }

  return { kind: 'reject' }
}

module.exports = { classifyOpenTarget }
