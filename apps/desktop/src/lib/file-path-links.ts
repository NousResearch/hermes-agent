// Detect bare file paths in assistant chat prose and turn them into clickable
// links. The desktop renderer routes these through MarkdownLink → FilePathLink:
// left-click opens the file in the in-app sandboxed preview pane; the
// right-click menu offers "Open in default app" / "Reveal in folder" through a
// hardened main-process IPC (see electron/hardening.cjs).
//
// Scope (v1) is deliberately narrow to keep false positives near zero:
//   - ABSOLUTE paths only — POSIX (`/…`, `~/…`) and Windows drive (`C:\…`,
//     `C:/…`). Relative paths (`src/app/x.tsx`) are NOT matched because, without
//     the session cwd, prose like "and/or" or "n/a" is too easy to mis-link.
//   - the path must end in a real filename WITH an extension; directories and
//     extensionless names (`/usr/local/bin`, `Makefile`) are skipped, and so
//     are dotfiles whose only dot is the leading one (`.env`, `.gitignore`).
//   - an optional `:line[:col]` suffix is kept in the visible label.
//
// Detection runs on prose only: fenced and inline code are split out upstream
// (markdown-preprocess.ts), and existing markdown links / autolinks are skipped
// here so we never rewrite a target that is already a link.

const FILE_PATH_HREF_PREFIX = '#file/'

// A single path-segment character: anything but whitespace, the separators, a
// colon (so a trailing `:line` is not swallowed and the Windows drive colon
// stays at the front), wildcards, quotes, angle brackets, and pipe.
const SEGMENT = '[^\\s/\\\\:*?"<>|]'
const LINE_SUFFIX = '(?::\\d+(?::\\d+)?)?'
// Extension: a dot + up to 16 alnum chars (covers `.properties`, `.storyboard`,
// `.entitlements`, ...). The END_BOUNDARY below makes an over-long or
// punctuation-continued tail (e.g. `.c-d`) fail the whole match cleanly instead
// of truncating to a wrong path.
const EXTENSION = '\\.[A-Za-z0-9]{1,16}'

// The matched path must END at a real boundary: whitespace, a path separator,
// quote/pipe/wildcard, or trailing sentence punctuation. This EXCLUDES
// segment-continuation chars (alnum, `-`, `_`), so the match can never stop in
// the middle of a filename — an over-long extension fails the match rather than
// truncating it. The class also ends a path before a closing quote (U+0027)
// or backtick (U+0060), listed by code point to avoid source-quoting issues.
const END_BOUNDARY = '(?=[\\s/\\\\:*?"<>|.,;!?)\\]}\\u0027\\u0060]|$)'

// The char immediately before a match must NOT be a word char, dot, tilde,
// colon, or a separator. This anchors matches to true absolute paths and skips
// relative `a/b.ts`, scheme-prefixed `file:/…`/`http:/…`, and the inner slashes
// of a longer token.
const PATH_BOUNDARY = '(?<![\\w.~:/\\\\])'

const POSIX_PATH = `~?(?:/${SEGMENT}+)+${EXTENSION}`
const WINDOWS_PATH = `[A-Za-z]:[\\\\/](?:${SEGMENT}+[\\\\/])*${SEGMENT}+${EXTENSION}`

const FILE_PATH_RE = new RegExp(`${PATH_BOUNDARY}(?:${WINDOWS_PATH}|${POSIX_PATH})${LINE_SUFFIX}${END_BOUNDARY}`, 'g')

// Anchored variant: matches a string that is EXACTLY one absolute file path
// (plus optional :line[:col]) and nothing else. Used to linkify a path the
// agent wrapped in inline code / a one-line code block, without touching code
// that merely contains a path (commands, snippets, flags).
const WHOLE_FILE_PATH_RE = new RegExp(`^(?:${WINDOWS_PATH}|${POSIX_PATH})${LINE_SUFFIX}$`)

// Existing markdown links `[text](target)` and autolinks `<…>`. Captured so
// linkification leaves them untouched and only rewrites the plain-text gaps.
// The destination class allows one level of balanced parens so a link target
// like `/build(1)/out.json` (or a `#preview/path(x)/…` href) is captured whole
// instead of being split at the first `)` and re-linkified into broken markup.
const MARKDOWN_LINK_OR_AUTOLINK_RE = /(\[[^\]]*\]\([^()]*(?:\([^()]*\)[^()]*)*\)|<[^\s>]+>)/g

const LINE_SUFFIX_RE = /:\d+(?::\d+)?$/

export interface FilePathParts {
  /** The full matched token, e.g. `/Users/me/out.ts:42` — shown as the label. */
  display: string
  /** The path with any `:line[:col]` suffix stripped — used to open/preview. */
  path: string
}

export function splitFilePathSuffix(value: string): FilePathParts {
  const path = value.replace(LINE_SUFFIX_RE, '')

  return { display: value, path: path || value }
}

export function filePathMarkdownHref(value: string): string {
  // encodeURIComponent leaves `(` and `)` unescaped, which would break the
  // markdown `(target)` wrapper — encode them too so the lexer keeps the href
  // intact.
  const encoded = encodeURIComponent(value).replace(/\(/g, '%28').replace(/\)/g, '%29')

  return `${FILE_PATH_HREF_PREFIX}${encoded}`
}

export function filePathFromMarkdownHref(href?: null | string): null | string {
  if (!href?.startsWith(FILE_PATH_HREF_PREFIX)) {
    return null
  }

  try {
    return decodeURIComponent(href.slice(FILE_PATH_HREF_PREFIX.length))
  } catch {
    return null
  }
}

// Only `[` and `]` can break the `[label]` wrapper. The visible label is
// re-rendered from the decoded href by FilePathLink, so this escaping only
// matters as a plain-markdown fallback.
function escapeMarkdownLinkLabel(value: string): string {
  return value.replace(/[[\]]/g, '\\$&')
}

// Build the `[label](#file/…)` markdown a detected path becomes.
export function fileLinkMarkdown(value: string): string {
  return `[${escapeMarkdownLinkLabel(value)}](${filePathMarkdownHref(value)})`
}

// True when `value` is exactly one absolute file path. Lets callers linkify a
// path the agent wrapped in inline code / a one-line code block while leaving
// code that only contains a path (e.g. `cat /a/b.ts`) untouched.
export function isLoneFilePath(value: string): boolean {
  return WHOLE_FILE_PATH_RE.test(value.trim())
}

export function linkifyFilePaths(text: string): string {
  if (!text.includes('/') && !/[A-Za-z]:[\\/]/.test(text)) {
    return text
  }

  return text
    .split(MARKDOWN_LINK_OR_AUTOLINK_RE)
    .map((part, index) => {
      // Odd indices are the captured links/autolinks — leave them untouched.
      if (index % 2 === 1) {
        return part
      }

      return part.replace(FILE_PATH_RE, match => fileLinkMarkdown(match))
    })
    .join('')
}
