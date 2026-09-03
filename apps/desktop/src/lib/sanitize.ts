// Format enforcers for identifier-style inputs, applied live (per keystroke) via
// <SanitizedInput>. They're intentionally lenient on a trailing separator so a
// value stays typeable (e.g. "feat/" then keep going); the final trim happens on
// submit / in the backend.

/** A git-ref-safe branch name: spaces → "-", drop chars git forbids, keep "/". */
export const gitRef = (raw: string): string =>
  raw
    .replace(/\s+/g, '-')
    .replace(/[^\w./-]/g, '') // \w = [A-Za-z0-9_]
    .replace(/-{2,}/g, '-')
    .replace(/\/{2,}/g, '/')
    .replace(/\.{2,}/g, '.')
    .replace(/^[-./]+/, '')

/** A profile identifier: lowercase, preserve underscores, normalize other separators to "-". */
export const slug = (raw: string): string =>
  raw
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '-')
    .replace(/^[-_]+/, '')
