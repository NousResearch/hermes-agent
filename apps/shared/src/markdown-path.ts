/**
 * Markdown/UI wrapper stripping for file paths taken from chat content.
 *
 * Assistant messages often wrap a file name in emphasis — `**Open xyz.pdf**`,
 * `__report final.docx__` — or in a composer directive prefix (`@url:`). When
 * such a string is used as the display label AND as the on-disk target, the
 * markdown syntax leaks into the path handed to the OS open handler and
 * Windows rejects it with "Windows cannot find '...xyz.pdf**'" (issue #95713).
 *
 * `stripMarkdownPathWrappers` unwraps those decorations before a path is
 * resolved or handed to `shell.openPath`. Only EDGES are touched: characters
 * in the middle of the path are never removed, and a string that unwraps to
 * nothing is returned unchanged.
 *
 * Consumed by the renderer (markdown link/image hrefs in
 * components/assistant-ui/markdown-text.tsx) and by the Electron main process
 * (openExternalUrl before shell.openPath). Keep both call sites wired — the
 * renderer fix keeps the display/preview pipeline clean, the main-process fix
 * is the last line of defense for the OS open handler.
 */

/** A `@url:`-style composer directive prefix (issue #95713). */
const DIRECTIVE_PREFIX_RE = /^@url:\s*/i

/**
 * Balanced emphasis pairs around the whole string. `__…__` / `_…_` only
 * unwrap when the inner text is NOT a bare word, so dunder-style names that
 * legitimately start and end with underscores (`__init__`, `__pycache__`,
 * `__main__`) survive untouched.
 */
const BALANCED_PAIRS: Array<{ re: RegExp; unwrapGuard?: (inner: string) => boolean }> = [
  { re: /^\*\*([\s\S]+)\*\*$/ },
  { re: /^__([\s\S]+)__$/, unwrapGuard: inner => /^[\w.-]+$/.test(inner) },
  { re: /^\*([^*]+)\*$/ },
  { re: /^_([^_]+)_$/, unwrapGuard: inner => /^[\w.-]+$/.test(inner) },
  { re: /^`([\s\S]+)`$/ }
]

export function stripMarkdownPathWrappers(raw: string): string {
  let value = String(raw ?? '').trim()

  if (!value) {
    return value
  }

  const original = value

  // A path can carry several layers (`**@url:file.pdf**`), so unwrap until
  // stable with a bounded number of passes.
  for (let pass = 0; pass < 8; pass += 1) {
    const before = value

    let matched = true

    while (matched) {
      matched = false

      const directive = DIRECTIVE_PREFIX_RE.exec(value)

      if (directive && value.length > directive[0].length) {
        value = value.slice(directive[0].length).trim()
        matched = true
      }

      for (const pair of BALANCED_PAIRS) {
        const match = pair.re.exec(value)

        if (!match) {
          continue
        }

        const inner = match[1] ?? ''

        // Blank inner, or a dunder-style word guarded against unwrapping:
        // leave this layer alone.
        if (!inner.trim() || pair.unwrapGuard?.(inner.trim())) {
          continue
        }

        value = inner.trim()
        matched = true
      }
    }

    if (value === before) {
      break
    }
  }

  // Stray unmatched decoration on the edges (`xyz.pdf**`, `**/logs/*.pdf`).
  // `*` and backticks are illegal in Windows file names and effectively never
  // appear at the edge of a real path; underscores stay — `report_final_` is
  // a perfectly legal name.
  value = value.replace(/^[*`]+/, '').replace(/[*`]+$/, '').trim()

  return value || original
}
