/**
 * Reading a `file:` URL back into the path it names. Desktop meets these URLs
 * in agent output, preview targets and the media pipeline, in the renderer and
 * in Electron main; every surface decodes them with this one rule so a URL
 * names the same file everywhere.
 */

// Drive letter after the path's leading slash. Node's win32 `fileURLToPath`
// tests the decoded path too, so `/C%3A/...` is a drive path.
const DRIVE_PATH_RE = /^\/[a-z]:/i

/**
 * The native path a `file:` URL names, read the way Node's `fileURLToPath`
 * reads it on the OS the URL itself implies: a UNC host or a drive letter can
 * only be Windows (`\\host\share\...`, `C:\...`), anything else is POSIX.
 *
 * The URL decides, not the OS this code runs on: the file may live on a remote
 * gateway or a WSL backend, so a host-platform reading would refuse the POSIX
 * URLs a Windows desktop receives from them, and a key derived from the path
 * would differ by desktop OS.
 *
 * Null when the value is not a parseable `file:` URL, does not percent-decode,
 * or encodes a separator: `%2f` anywhere, `%5c` in a Windows path (a POSIX
 * name may hold a literal backslash).
 */
export function fileUrlToNativePath(value: string): null | string {
  try {
    const { hostname, pathname, protocol } = new URL(value)

    // Decoding an encoded separator would add a segment the URL parser never normalized.
    if (protocol !== 'file:' || /%2f/i.test(pathname)) {
      return null
    }

    const path = decodeURIComponent(pathname)

    if (!hostname && !DRIVE_PATH_RE.test(path)) {
      return path
    }

    if (/%5c/i.test(pathname)) {
      return null
    }

    return (hostname ? `//${hostname}${path}` : path.slice(1)).replace(/\//g, '\\')
  } catch {
    return null
  }
}
