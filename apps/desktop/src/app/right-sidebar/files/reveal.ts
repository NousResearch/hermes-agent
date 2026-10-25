/**
 * Revealing a file in the project tree.
 *
 * Pure and path-only so the behaviour is testable without a rendered tree: the
 * component owns scrolling and selection, this owns which folders must be open
 * for a path to be reachable at all.
 */

/** POSIX and Windows separators, since a remote workspace may be either. */
function splitPath(path: string): { sep: string; parts: string[] } {
  const sep = path.includes('\\') && !path.includes('/') ? '\\' : '/'
  return { parts: path.split(sep).filter(Boolean), sep }
}

/**
 * Every ancestor folder of `path` inside `root`, outermost first.
 *
 * Outermost first matters: a tree that loads children lazily can only resolve
 * `a/b/c` after `a/b` exists, so opening in this order is what makes a deep
 * path reachable in one pass.
 *
 * `path` outside `root` yields nothing — a file from another workspace has no
 * place in this tree, and guessing one would scroll to an unrelated row.
 */
export function ancestorPaths(path: string, root: string): string[] {
  if (!path || !root) return []
  const normalizedRoot = root.replace(/[/\\]+$/, '')
  if (path === normalizedRoot) return []
  const rootWithSep = normalizedRoot + (normalizedRoot.includes('\\') ? '\\' : '/')
  if (!path.startsWith(rootWithSep)) return []

  const relative = path.slice(rootWithSep.length)
  const { parts, sep } = splitPath(relative)
  // The last part is the file itself; only its ancestors need opening.
  const folders = parts.slice(0, -1)

  const result: string[] = []
  let current = normalizedRoot
  for (const part of folders) {
    current = current + sep + part
    result.push(current)
  }
  return result
}

/**
 * The open-state patch that reveals `path`.
 *
 * Only ever ADDS open folders. Revealing a file must not collapse the branches
 * the user opened to get somewhere else — a reveal is an offer, not a reset.
 */
export function openStateForReveal(
  path: string,
  root: string,
  current: Record<string, boolean>
): Record<string, boolean> {
  const needed = ancestorPaths(path, root)
  if (needed.length === 0) return current

  let changed = false
  const next = { ...current }
  for (const folder of needed) {
    if (!next[folder]) {
      next[folder] = true
      changed = true
    }
  }
  // Preserve reference identity when nothing moved: the tree re-renders on a
  // new object, and an already-visible file is the common case.
  return changed ? next : current
}
