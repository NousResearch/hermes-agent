// `file://` URL for an ABSOLUTE local path, or null when the path has no URL
// form without a base directory (relative, `~`). Pure so the renderer (which
// builds the URL) and the Electron main tests (which resolve it the way
// `resolveRequestedPathForIpc` does) share one implementation.
//
// String concatenation (`file://${path}`) is not a file URL: `#` and `?` start
// the fragment/query, so `/tmp/a#b.png` opened `/tmp/a`; a relative name became
// the URL host (`file://out.png`), which the main process rejects. Each segment
// is percent-encoded instead so spaces, `#`, `?`, `%` and non-ASCII round-trip
// through `fileURLToPath`.

const encodeSegments = (path: string) => path.split('/').map(encodeURIComponent).join('/')

export function localFileUrl(path: string): null | string {
  if (/^file:/i.test(path)) {
    return path
  }

  // Windows drive path: `C:\x` / `C:/x` -> `file:///C:/x` (colon kept literal).
  const drive = /^([a-zA-Z]):[\\/](.*)$/.exec(path)

  if (drive) {
    return `file:///${drive[1]}:/${encodeSegments(drive[2].replace(/\\/g, '/'))}`
  }

  // UNC path: `\\server\share\x` -> `file://server/share/x`.
  const unc = /^\\\\([^\\/]+)[\\/](.*)$/.exec(path)

  if (unc) {
    return `file://${unc[1]}/${encodeSegments(unc[2].replace(/\\/g, '/'))}`
  }

  // POSIX absolute. Backslashes are legal filename characters here, so they
  // are encoded rather than treated as separators.
  if (path.startsWith('/')) {
    return `file://${encodeSegments(path)}`
  }

  return null
}
