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

// 相对路径必须包含至少一个目录分隔符（./ ../ 或 /），这样
// seed-audio-1.0 / whisper-large-v3 等技术词不会被当成路径。
// 两种形式：
//   1) 以 ./ 或 ../ 开头：./src/main.ts, ../shared/types.ts
//   2) 有至少一个 / 分隔：docs/guide.md, a/b/c/file.ts
// 匹配非贪婪以避免吞相邻路径。
// 扩展名限 1-8 个字母数字（.ts, .py, .json, .md 等），不含点号以免
// seed-audio-1.0 中 .0 被当扩展名。
const RELATIVE_FILE_PATH =
  /(?<![A-Za-z0-9_/.])((?:\.\.?\/[\w@%.-]+(?:\/[\w@%.-]+)*|[\w@%.-]+\/[\w@%.-]+(?:\/[\w@%.-]+)*)\.([A-Za-z0-9]{1,8}))(?![A-Za-z0-9_/])/g

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
  // Mask existing markdown links ([label](url)) and inline code (`…`) with
  // placeholders so their targets are never relinked, then restore after
  // linking plain paths. Inline code must be protected too: a path inside
  // backticks renders as code, not as a link, so rewriting it there would
  // leak raw markdown syntax onto the screen.
  const protectedSpans: string[] = []
  const mask = (match: string) => {
    protectedSpans.push(match)
    return `\u0000${protectedSpans.length - 1}\u0000`
  }
  // Protect spans that must never be re-linked into media attachments:
  // existing markdown links, inline code, and emphasis/strikethrough markers
  // (bold `**x**`, `__x__`; italic `*x*`, `_x_`; strikethrough `~~x~~`). A
  // path wrapped in emphasis is prose the author chose to highlight, not a
  // file they're pointing at — relinking it (e.g. turning `**docs/README.md**`
  // into an "Open README.md" media chip) is wrong. `_x_`/`*x*` need a
  // non-word boundary on both sides so arithmetic (`2*3`, `a_b`) isn't
  // mistaken for italic.
  const masked = chunk
    .replace(/\[[^\]]*\]\([^)]*\)/g, mask)
    .replace(/`[^`\n]+`/g, mask)
    .replace(/\*\*[^*\n]+\*\*/g, mask)
    .replace(/__[^_\n]+__/g, mask)
    .replace(/~~[^~\n]+~~/g, mask)
    .replace(/(?<!\w)[*_][^\n*_]+[*_](?!\w)/g, mask)

  let linked = masked.replace(ABSOLUTE_FILE_PATH, (path: string) => `[${path}](#media:${encodeURIComponent(path)})`)

  if (cwd) {
    linked = linked.replace(RELATIVE_FILE_PATH, (rel: string) => {
      const absolute = resolveRelativePath(cwd, rel)
      return `[${rel}](#media:${encodeURIComponent(absolute)})`
    })
  }

  // Restore masked spans. Masks can nest (a bold span wrapping an inline-code
  // span: `**\`hermes update\`**`), so a single replace pass would leave the
  // inner placeholder behind — the restored outer text still contains
  // \u0000N\u0000, which renders as the U+FFFD replacement character on
  // screen. Loop until no placeholder remains.
  let restored = linked
  for (let guard = 0; guard < protectedSpans.length + 1; guard += 1) {
    const next = restored.replace(/\u0000(\d+)\u0000/g, (_match, index: string) => protectedSpans[Number(index)] ?? '')
    if (next === restored) {
      break
    }
    restored = next
  }
  return restored
}

/** Resolve a relative path (`./`, `../`, plain segments) against `cwd`. */
function resolveRelativePath(cwd: string, rel: string): string {
  const isAbsolute = cwd.startsWith('/')
  const stack = cwd.split('/')
  for (const segment of rel.split('/')) {
    if (segment === '' || segment === '.') {
      continue
    }
    if (segment === '..') {
      // Clamp at the filesystem root: `../../..` against a shallow cwd must
      // not pop past the root and produce a path relative to the app's own
      // cwd. `Math.max(0, ...)` keeps the stack non-empty.
      stack.splice(Math.max(0, stack.length - 1), 1)
    } else {
      stack.push(segment)
    }
  }
  let joined = stack.join('/')
  // Preserve the leading root slash for absolute cwds (split('/') on
  // '/repo' yields ['', 'repo'], and popping past 'repo' must not drop the
  // root marker).
  if (isAbsolute && !joined.startsWith('/')) {
    joined = '/' + joined
  }
  return joined
}
