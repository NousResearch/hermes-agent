/**
 * Resolve inline (cwd-relative) and pill (absolute) attachment paths to the same
 * equivalence key so merge/dedupe treats them as one ref.
 */
const ABSOLUTE_OS_PATH_RE = /^([A-Za-z]:[\\/]|\/)/

export function isAbsoluteOsPath(path: string): boolean {
  return ABSOLUTE_OS_PATH_RE.test(path.trim())
}

function joinPath(base: string, rel: string): string {
  if (!base) {
    return rel
  }

  return `${base.replace(/\/+$/, '')}/${rel.replace(/^\.?\//, '')}`
}

function normalizeSlashes(path: string): string {
  return path.replace(/\\/g, '/')
}

export function resolveAttachmentPath(path: string, cwd: string): string {
  const trimmed = path.trim()

  if (!trimmed) {
    return trimmed
  }

  if (isAbsoluteOsPath(trimmed)) {
    return normalizeSlashes(trimmed)
  }

  if (cwd) {
    return normalizeSlashes(joinPath(cwd, trimmed))
  }

  return normalizeSlashes(trimmed)
}

/** True when two paths refer to the same attachment (absolute vs cwd-relative tail). */
export function attachmentPathsEquivalent(pathA: string, pathB: string, cwd = ''): boolean {
  const a = normalizeSlashes(pathA.trim())
  const b = normalizeSlashes(pathB.trim())

  if (!a || !b) {
    return false
  }

  if (a === b) {
    return true
  }

  if (resolveAttachmentPath(a, cwd) === resolveAttachmentPath(b, cwd)) {
    return true
  }

  if (isAbsoluteOsPath(a) && !isAbsoluteOsPath(b)) {
    return a.endsWith(`/${b}`) || a.endsWith(b)
  }

  if (isAbsoluteOsPath(b) && !isAbsoluteOsPath(a)) {
    return b.endsWith(`/${a}`) || b.endsWith(a)
  }

  return false
}

export function attachmentEquivalenceKey(kind: string, path: string, cwd = ''): string {
  return `${kind}:${resolveAttachmentPath(path, cwd)}`
}

/** Prefer absolute OS paths when two tokens refer to the same attachment. */
export function preferCanonicalAttachmentPath(existing: string, incoming: string): string {
  if (isAbsoluteOsPath(incoming)) {
    return normalizeSlashes(incoming)
  }

  if (isAbsoluteOsPath(existing)) {
    return normalizeSlashes(existing)
  }

  return normalizeSlashes(incoming)
}
