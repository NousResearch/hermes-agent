/**
 * Linkify plain-text file paths in markdown source so they become clickable
 * links that open the file with the OS default application (via the existing
 * `#media:` attachment mechanism in MarkdownLink).
 *
 * Opt-in only (`desktop.markdown_linkify_paths`); off by default to preserve
 * historical rendering. Absolute paths are always matched; relative paths are
 * matched only when a `cwd` is provided (they resolve against it). Code
 * fences and existing markdown links are left untouched.
 */

// Absolute path with a file extension, e.g. /Users/echo/notes.md,
// /tmp/build.log, /app/src/main.ts. Non-ASCII (e.g. Chinese) file names are
// allowed but whitespace is not: paths in prose are token-like, and allowing
// spaces makes consecutive paths (`/tmp/a.ts 与 /tmp/b.json`) or a bare
// directory + trailing text (`/Users/echo/notes 与 .../README.md`) swallow
// into one link. The match is non-greedy so consecutive paths each link
// separately; the trailing `(?![A-Za-z0-9_])` stops truncating a match right
// before a suffix (e.g. doc.md). The leading lookbehind also rejects a dot
// or slash before the match so `./src/main.ts` stays relative and URL hosts
// (`https://github.com/…`) never get linked as local files. The `(?!\/)` also
// rejects double slashes — a URL scheme's `//` must not start a match.
const ABSOLUTE_FILE_PATH = /(?<![A-Za-z0-9_./])(\/(?!\/)[^`"<>\[\]{}()\s]+?\.([A-Za-z0-9]{1,8}))(?![A-Za-z0-9_])/g

// Project-relative path with an extension: docs/guide.md, ./src/main.ts,
// ../shared/types.ts. The leading lookbehind rejects URLs (path segment right
// after a dot like github.com/user/file.md) and any segment glued to a word
// character; the trailing lookbehind avoids half-path truncation.
const RELATIVE_FILE_PATH =
  /(?<![A-Za-z0-9_/.])((?:\.\.?\/)?[\w@%.-]+(?:\/[\w@%.-]+)*\.(?:[A-Za-z0-9]{1,8}))(?![A-Za-z0-9_/])/g

/**
 * Turn file paths in markdown text into `#media:` links.
 * Code fences (``` blocks) and existing `[label](url)` links are preserved.
 * Absolute paths link unconditionally; relative paths link only when `cwd`
 * is provided (they resolve against it, so the click opens the real file).
 */
export function linkifyFilePaths(source: string, cwd?: string): string {
  // Split on code-fence markers; even indexes are prose, odd indexes are
  // inside a fence (```lang ... ```). The assistant pipeline emits backticks.
  const parts = source.split(/```/)
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = linkifyProseChunk(parts[i], cwd)
  }
  return parts.join('```')
}

function linkifyProseChunk(chunk: string, cwd?: string): string {
  // Mask existing markdown links ([label](url)) with placeholders so their
  // targets are never relinked, then restore after linking plain paths.
  const protectedSpans: string[] = []
  const masked = chunk.replace(/\[[^\]]*\]\([^)]*\)/g, match => {
    protectedSpans.push(match)
    return `\u0000${protectedSpans.length - 1}\u0000`
  })

  let linked = masked.replace(ABSOLUTE_FILE_PATH, (path: string) => `[${path}](#media:${encodeURIComponent(path)})`)

  if (cwd) {
    linked = linked.replace(RELATIVE_FILE_PATH, (rel: string) => {
      const absolute = resolveRelativePath(cwd, rel)
      return `[${rel}](#media:${encodeURIComponent(absolute)})`
    })
  }

  return linked.replace(/\u0000(\d+)\u0000/g, (_match, index: string) => protectedSpans[Number(index)])
}

/** Resolve a relative path (`./`, `../`, plain segments) against `cwd`. */
function resolveRelativePath(cwd: string, rel: string): string {
  const stack = cwd.split('/')
  for (const segment of rel.split('/')) {
    if (segment === '' || segment === '.') {
      continue
    }
    if (segment === '..') {
      stack.pop()
    } else {
      stack.push(segment)
    }
  }
  return stack.join('/')
}
