import { host } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

/**
 * Linkify absolute filesystem paths inside a kanban comment body.
 *
 * When a worker mentions a file (e.g. "Updated `/opt/data/skills/x/SKILL.md` to
 * v1.4.1"), the plain path is a dead end — the reviewer has to hunt the file
 * browser manually. This renders recognized absolute paths as distinct-styled,
 * clickable affordances that reveal + select the file in the workspace tree via
 * `host.revealFileInTree`, so review → open file is one click.
 *
 * Only ABSOLUTE paths are matched (POSIX `/…` and Windows `C:\…` / `C:/…` /
 * UNC `\\server\share\…`), and the reveal door is workspace-scoped — the tree
 * ignores paths outside the active workspace cwd — so arbitrary system paths
 * never become clickable.
 */

// One run of path characters: letters, digits, `_`, `.`, `~`, `+`, `@`, `/` and
// a literal backslash, with `-` last. Deliberately excludes sentence/delimiter
// punctuation and quotes so prose like "see /a/b.txt!" or "(/c/d.ts)" stops
// exactly at the path, not mid-filename. A trailing `/` (directory reference)
// is allowed.
const PATH_RUN = "[A-Za-z0-9_.~+@/\\\\-]+"

// Anchor candidates at a boundary: start of string, or after whitespace /
// `(` / `[` / backtick / single or double quote.
const BOUNDARY = "(?:^|(?<=[\\s(\\[{`\"']))"

// POSIX absolute path: `/` + a path-char run. The negative lookahead after the
// opening `/` rejects a bare `/` and a leading `//`.
const POSIX_ABS_PATH_RE = new RegExp(BOUNDARY + "/(?!/)" + PATH_RUN, 'g')

// Windows drive path: `C:` + `/` or `\` + a path-char run.
const WINDOWS_DRIVE_RE = new RegExp(BOUNDARY + "[A-Za-z]:[/\\]" + PATH_RUN, 'g')

// Windows UNC path: `\\server\share\…` — two backslashes, a server name, a
// backslash, a share name, then a path-char run.
const WINDOWS_UNC_RE = new RegExp(BOUNDARY + "\\\\[^\\/\\s]+\\\\[^\\/\\s]+" + PATH_RUN, 'g')

const ABS_PATH_RE = new RegExp(
  `(?:${POSIX_ABS_PATH_RE.source})|(?:${WINDOWS_DRIVE_RE.source})|(?:${WINDOWS_UNC_RE.source})`,
  'g'
)

/** Does `value` look like a filesystem path a reviewer would want to open? */
export function isAbsoluteFilePath(value: string): boolean {
  const trimmed = (value || '').trim()

  return (
    (trimmed.startsWith('/') && trimmed.length > 1) ||
    /^[A-Za-z]:[\\/]/.test(trimmed) ||
    /^\\\\[^\\/]+\\[^\\/]+/.test(trimmed)
  )
}

interface LinkifiedFilePathProps {
  text: string
}

/**
 * Render a comment body with absolute file paths turned into clickable links.
 * Every other token stays literal text (no markdown interpretation), matching
 * the plain-text rendering kanban comments use today.
 */
export function LinkifiedFilePath({ text }: LinkifiedFilePathProps): ReactNode {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of text.matchAll(ABS_PATH_RE)) {
    const raw = match[0]
    const index = match.index ?? 0

    if (index > cursor) {
      nodes.push(text.slice(cursor, index))
    }

    // A trailing period from sentence punctuation (e.g. "…file.tar.gz.") is not
    // part of a filename; drop it before linking.
    const path = raw.replace(/[.]+$/, '')

    if (path && isAbsoluteFilePath(path)) {
      nodes.push(
        <button
          aria-label={`Reveal ${path} in file tree`}
          className="ref kanban-filepath-link cursor-pointer"
          key={`${path}-${index}`}
          onClick={() => host.revealFileInTree(path)}
          title={path}
          type="button"
        >
          {path}
        </button>
      )
      cursor = index + raw.length
    } else {
      // Not a linkable path — keep the raw match literal.
      nodes.push(text.slice(index, index + raw.length))
      cursor = index + raw.length
    }
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor))
  }

  return nodes.length ? <>{nodes}</> : text
}
