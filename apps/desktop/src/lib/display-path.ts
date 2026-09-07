/**
 * One place to format filesystem paths for DISPLAY.
 *
 * Industry standard (shells, VS Code `tildify`, Finder path bar): collapse the
 * user's home directory to `~`, keep forward slashes, leave everything else
 * alone. Copy/reveal/IPC still use the real absolute path — this is paint only.
 *
 * When `home` is unknown (renderer has no `os.homedir()` and remote cwd may
 * not match the local machine), a conservative heuristic still collapses the
 * common `/Users/<name>`, `/home/<name>`, and `C:/Users/<name>` prefixes so a
 * long absolute path never paints raw in chrome.
 */

export interface DisplayPathOptions {
  /** Explicit home directory to collapse (local machine home, remote $HOME). */
  home?: null | string
}

/** Normalize separators and drop a trailing slash (except root / drive root). */
export function normalizeDisplayPath(raw: string): string {
  let path = (raw || '').trim().replace(/\\/g, '/')

  if (!path) {
    return ''
  }

  // Collapse repeated slashes, but keep a leading UNC `//server/...` pair.
  if (path.startsWith('//')) {
    path = `//${path.slice(2).replace(/\/{2,}/g, '/')}`
  } else {
    path = path.replace(/\/{2,}/g, '/')
  }

  // Drop trailing slash except bare `/` or `C:/`.
  if (path.length > 1 && path.endsWith('/') && !/^[A-Za-z]:\/$/.test(path)) {
    path = path.replace(/\/+$/, '')
  }

  return path
}

function normalizeHome(home: string): string {
  const normalized = normalizeDisplayPath(home)

  if (!normalized) {
    return ''
  }

  // Home itself should not keep a trailing slash for prefix checks.
  return normalized.replace(/\/+$/, '')
}

function startsWithHome(path: string, home: string, caseInsensitive: boolean): boolean {
  if (!home) {
    return false
  }

  if (path === home) {
    return true
  }

  const prefix = `${home}/`

  return caseInsensitive
    ? path.toLowerCase().startsWith(prefix.toLowerCase()) || path.toLowerCase() === home.toLowerCase()
    : path.startsWith(prefix) || path === home
}

/**
 * Best-effort home prefix when callers don't pass one. Matches the usual
 * single-user layouts; never collapses `/Users` or `/home` alone.
 */
function inferredHomePrefix(path: string): string {
  // macOS: /Users/name[/...]
  let match = path.match(/^(\/Users\/[^/]+)(?:\/|$)/)

  if (match) {
    return match[1]
  }

  // Linux (and most UNIX): /home/name[/...]
  match = path.match(/^(\/home\/[^/]+)(?:\/|$)/)

  if (match) {
    return match[1]
  }

  // Windows user profile: C:/Users/name[/...] (also works after \ → /)
  match = path.match(/^([A-Za-z]:\/Users\/[^/]+)(?:\/|$)/)

  if (match) {
    return match[1]
  }

  return ''
}

/**
 * Format a filesystem path for UI chrome.
 *
 *   /Users/brooklyn/www/hermes-agent  →  ~/www/hermes-agent
 *   /Users/brooklyn                   →  ~
 *   C:\Users\brooklyn\src             →  ~/src
 *   /var/log                          →  /var/log
 *   already/relative                  →  already/relative
 */
export function displayPath(raw: null | string | undefined, options: DisplayPathOptions = {}): string {
  const path = normalizeDisplayPath(raw || '')

  if (!path) {
    return ''
  }

  // Already tildified — normalize only.
  if (path === '~' || path.startsWith('~/')) {
    return path
  }

  const explicitHome = options.home ? normalizeHome(options.home) : ''
  // Windows paths are case-insensitive; POSIX paths with an explicit home keep
  // case-sensitive matching (Linux). Heuristic homes on mac/win ignore case.
  const home = explicitHome || inferredHomePrefix(path)

  if (!home) {
    return path
  }

  const caseInsensitive = !explicitHome || /^[A-Za-z]:\//.test(home) || home.startsWith('/Users/')

  if (!startsWithHome(path, home, caseInsensitive)) {
    return path
  }

  if (path.length === home.length) {
    return '~'
  }

  return `~${path.slice(home.length)}`
}

/** Last path segment for compact labels (statusbar leaf, settings rows). */
export function pathLeaf(raw: null | string | undefined): string {
  const path = normalizeDisplayPath(raw || '')

  if (!path || path === '/' || path === '~') {
    return path
  }

  // `C:/` drive root
  if (/^[A-Za-z]:\/$/.test(path) || /^[A-Za-z]:$/.test(path)) {
    return path.endsWith('/') ? path : `${path}/`
  }

  const leaf = path.split('/').filter(Boolean).pop()

  return leaf || path
}
