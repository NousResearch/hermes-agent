/**
 * Markdown hrefs percent-encode spaces (`/my%20notes/x.md`), but the on-disk
 * path is the decoded form. When a preview resolve misses, retry the decoded
 * path so chat links to paths with spaces still open (#102782).
 */
export function decodedPathIfMissing(resolvedPath: string): null | string {
  if (!/%[0-9a-fA-F]{2}/.test(resolvedPath)) {
    return null
  }

  let decoded: string

  try {
    decoded = decodeURIComponent(resolvedPath)
  } catch {
    return null
  }

  return decoded !== resolvedPath ? decoded : null
}
