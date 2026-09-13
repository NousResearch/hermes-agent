/** Comparing two paths for identity or containment, across host spellings.
 *
 *  A cwd reaches us from the backend, a picker, a session row and a project
 *  record, and the four don't agree on separator, trailing slash or drive-letter
 *  case. Compare through these rather than `===` / `startsWith`.
 */

/** POSIX-style spelling: one separator, no trailing slash.
 *
 *  Repeated separators collapse as well: `C:/Repos//App` names the same
 *  directory as `C:/Repos/App`, and a cwd assembled by joining a root that
 *  already ends in a separator arrives spelled that way. A UNC path keeps its
 *  leading `//`, the one place a doubled separator carries meaning. */
export const cleanPath = (path: string): string => {
  const unified = path.trim().replace(/\\/g, '/')
  const uncPrefix = unified.startsWith('//') ? '/' : ''
  const collapsed = unified.replace(/\/{2,}/g, '/').replace(/\/+$/, '')

  return collapsed === '' ? '/' : `${uncPrefix}${collapsed}`
}

/** Case-folded comparison key. Windows drive/UNC paths are case-insensitive;
 *  POSIX paths are not, and callers that display a path want its real spelling,
 *  so fold only the key. Expects an already-`cleanPath`ed value. */
export const comparisonPath = (path: string): string =>
  /^[A-Za-z]:(?:\/|$)/.test(path) || path.startsWith('//') ? path.toLowerCase() : path

/** True when `child` IS `parent` or lives underneath it. */
export const isUnderPath = (parent: string, child: string): boolean => {
  const p = comparisonPath(cleanPath(parent))
  const c = comparisonPath(cleanPath(child))

  return c === p || c.startsWith(`${p}/`)
}
