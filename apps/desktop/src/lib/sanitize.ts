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

/** A kebab slug: lowercase, runs of non-alphanumerics → a single "-". */
export const slug = (raw: string): string =>
  raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+/, '')

// Invisible Unicode that renders as nothing but still lands in projects.db rows
// and folder names, where it silently breaks string comparison (session
// routing/grouping) and path matching: control, format (ZWSP/ZWNJ/ZWJ, bidi
// marks, BOM, soft hyphen), private-use, line/paragraph separators, and the
// trailing specials block. Surrogates can't occur here — an input's value is a
// USVString.
const INVISIBLE = /[\p{Cc}\p{Cf}\p{Co}\p{Zl}\p{Zp}\uFFF0-\uFFFF]/gu

/** A display-safe project name: invisible characters dropped as you type. */
export const projectName = (raw: string): string => raw.replace(INVISIBLE, '')
