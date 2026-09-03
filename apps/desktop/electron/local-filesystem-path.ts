// Pure path/URL predicates for the main-process opener. Kept out of main.ts
// so protocol-relative `//host/path` cannot silently become openPath.

const LOCAL_FILESYSTEM_PATH_RE = /^(?:\/(?!\/)|\/\/\/|~\/|[a-zA-Z]:[\\/]|\\\\)/
const PROTOCOL_RELATIVE_RE = /^\/\/[^/\s]/

/** POSIX `/…`, `~/…`, Windows drive/UNC (`\\server\…`). Not `//host/…`. */
export function looksLikeLocalFilesystemPath(value: string): boolean {
  return LOCAL_FILESYSTEM_PATH_RE.test(value)
}

/**
 * `//cdn.example.com/img.png` is a URL, not a filesystem path. `new URL()`
 * rejects it without a base; open it as https so it does not fall through
 * to `shell.openPath`. `///tmp/foo` stays a POSIX path.
 */
export function absolutizeProtocolRelativeUrl(value: string): string {
  if (PROTOCOL_RELATIVE_RE.test(value)) {
    return `https:${value}`
  }

  return value
}
