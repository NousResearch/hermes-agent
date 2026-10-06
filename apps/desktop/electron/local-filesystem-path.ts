// Pure path/URL predicates for the main-process opener. Kept out of
// external-open.ts so `classifyOpenTarget`-style decisions stay testable
// without electron, and so protocol-relative `//host/path` can never silently
// fall through to shell.openPath (it is a URL, not a filesystem path).
//
// A bare local path here is one that `new URL()` either rejects outright
// (POSIX `/…`, `~/…`, UNC `\\server\…`) or mis-parses as a bogus scheme
// (`C:\…` parses as protocol `c:`, which the web allowlist must never see).
// Chat/media links and artifact values arrive at the opener carrying exactly
// these shapes when the renderer hands a raw path through instead of a
// canonical `file://` URL (hermes-agent 80946, 84361).

const LOCAL_FILESYSTEM_PATH_RE = /^(?:\/(?!\/)|\/\/\/|~\/|[a-zA-Z]:[\\/]|\\\\)/
const PROTOCOL_RELATIVE_RE = /^\/\/[^/\s]/

/**
 * POSIX `/…`, `///…`, `~/…`, Windows drive (`C:\…`, `C:/…`) and UNC
 * (`\\server\…`) paths. NOT `//host/…` — that is a protocol-relative URL.
 */
export function looksLikeLocalFilesystemPath(value: string): boolean {
  return LOCAL_FILESYSTEM_PATH_RE.test(value)
}

/**
 * `//cdn.example.com/img.png` is a URL, not a filesystem path. `new URL()`
 * rejects it without a base; open it as `https:` so it reaches the system
 * browser instead of being treated as (or falling through to) a local path.
 * `///tmp/foo` stays a POSIX path.
 */
export function absolutizeProtocolRelativeUrl(value: string): string {
  if (PROTOCOL_RELATIVE_RE.test(value)) {
    return `https:${value}`
  }

  return value
}
