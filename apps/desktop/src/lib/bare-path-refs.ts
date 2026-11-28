/**
 * Bare filesystem paths and URLs in assistant prose (#48313).
 *
 * Agents hand over paths as plain text far more often than as markdown links:
 * "Report: /Users/me/report.md", or `` `/tmp/out.md` `` in a code span. An
 * authored `[report](/path)` already reaches the preview pipeline through
 * `routeFileLinksToPreview`; this finds the paths the agent did NOT wrap so
 * they can open through the same pipeline.
 *
 * Pure and free of React/store imports so `preprocessMarkdown` (a hot
 * per-flush path) can call it, like `session-refs.ts`.
 */

/** Extensions worth linking in prose. A list, not "any dot": `/usr/bin` or
 *  `a/b` in prose is more often a directory, a command or a ratio than a file
 *  to preview. Executables, installers and archives are left out on purpose:
 *  a link only ever opens the preview pane, but nothing the agent mentions in
 *  passing should look like a one-click launcher. */
const LINKABLE_EXTENSIONS = [
  'c',
  'cjs',
  'conf',
  'cpp',
  'css',
  'csv',
  'gif',
  'go',
  'h',
  'hpp',
  'htm',
  'html',
  'ini',
  'java',
  'jpeg',
  'jpg',
  'js',
  'json',
  'jsonl',
  'jsx',
  'log',
  'lua',
  'markdown',
  'md',
  'mdx',
  'mjs',
  'pdf',
  'png',
  'py',
  'rb',
  'rs',
  'sql',
  'svg',
  'toml',
  'ts',
  'tsx',
  'txt',
  'webp',
  'xml',
  'yaml',
  'yml'
] as const

/** Never linked, even when the agent puts the path in a code span: the click
 *  opens a preview, but a chip that reads as "open this program" is the wrong
 *  affordance and an easy social-engineering lure. */
const EXECUTABLE_EXTENSIONS = new Set([
  'app',
  'appimage',
  'bat',
  'cmd',
  'com',
  'command',
  'cpl',
  'deb',
  'dmg',
  'exe',
  'jar',
  'msi',
  'pkg',
  'ps1',
  'rpm',
  'scr',
  'tool',
  'vbs',
  'workflow'
])

const EXT_ALT = LINKABLE_EXTENSIONS.join('|')

/**
 * A bare absolute path in prose: `/abs/…`, `~/…` or `file://…`, ending in a
 * known extension. Path characters exclude whitespace and markdown-significant
 * punctuation, so a path never swallows a closing paren, bracket or quote.
 * Windows drive paths are out of scope: markdown reads `\` as an escape.
 *
 * The lookbehind rejects a path glued to a word (`docs/a.md` is relative), a
 * link target (`](/x.md)`), a URL path (`host:8000/a.md`, `//a.md`) and an
 * HTML attribute value (`href="/a.md"`).
 */
const BARE_PATH_RE = new RegExp(
  String.raw`(?<![\w\]\(<:/=~.\-]|=["'])(?:file:\/\/|~(?=\/)|(?=\/))[^\s[\]()<>"'\x60*]*?\.(?:${EXT_ALT})(?![\w\-/])(?!\.\w)`,
  'gi'
)

/** Angle-bracket spans (autolinks, inline HTML tags) own their contents. */
const ANGLE_SPAN_SPLIT_RE = /(<[^<>\n]*>)/

/** Sentence punctuation glued to the end of a path. */
const TRAILING_PUNCTUATION_RE = /[,.;:!?]+$/

const CODE_SPAN_URL_RE = /^https?:\/\/[^\s<>"'`]+$/i
const CODE_SPAN_PATH_RE = /^(?:file:\/\/|~\/|\/)[^\s<>"'`]+$/i

const PATH_HREF_PREFIX = '#path/'

function extensionOf(path: string): string {
  const name = path.split('/').pop() || ''
  const dot = name.lastIndexOf('.')

  return dot > 0 ? name.slice(dot + 1).toLowerCase() : ''
}

/** True when a path points at something executable we refuse to link. */
export function isExecutablePath(path: string): boolean {
  const clean = path.replace(/\/+$/, '')

  return EXECUTABLE_EXTENSIONS.has(extensionOf(clean))
}

/**
 * Classifies the text of an inline code span. `'url'` for an http(s) URL that
 * fills the span, `'path'` for an absolute, home or `file://` path, `null` for
 * everything else (a command, a symbol, a relative path). A code span is a
 * deliberate hand-over, so any extension qualifies (`/etc/hosts`), except
 * executables.
 */
export function classifyCodeSpan(text: string): 'path' | 'url' | null {
  const value = text.trim()

  if (!value || /\s/.test(value)) {
    return null
  }

  if (CODE_SPAN_URL_RE.test(value)) {
    return 'url'
  }

  if (CODE_SPAN_PATH_RE.test(value) && value !== '/' && !isExecutablePath(value)) {
    return 'path'
  }

  return null
}

/** The href a bare path takes: `#path/<encoded>`. Fragment hrefs survive
 *  markdown URL hardening, the same trick as `#session/` and `#preview/`. It
 *  is separate from `#preview/` because that door renders the block
 *  PreviewAttachment card for links the agent authored; a path mentioned
 *  mid-sentence stays an inline link so the sentence still reads. `~` is
 *  encoded so lone-tilde escaping cannot touch the href. */
export function pathMarkdownHref(path: string): string {
  return `${PATH_HREF_PREFIX}${encodeURIComponent(path).replace(/~/g, '%7E')}`
}

export function pathFromMarkdownHref(href?: string): null | string {
  if (!href?.startsWith(PATH_HREF_PREFIX)) {
    return null
  }

  try {
    return decodeURIComponent(href.slice(PATH_HREF_PREFIX.length)) || null
  } catch {
    return null
  }
}

function linkifySegment(text: string): string {
  return text.replace(BARE_PATH_RE, (match: string) => {
    const path = match.replace(TRAILING_PUNCTUATION_RE, '')
    const trailing = match.slice(path.length)

    if (!path || isExecutablePath(path)) {
      return match
    }

    return `[${path.replace(/[[\]\\]/g, '\\$&')}](${pathMarkdownHref(path)})${trailing}`
  })
}

/**
 * Rewrites bare absolute paths in prose into `#path/` markdown links. The
 * label is the path as written, so the reader still sees (and can copy) it.
 * Callers pass prose only: no code spans, fences, or text inside existing
 * markdown links (`preprocessMarkdown` splits those off first).
 */
export function linkifyBarePaths(text: string): string {
  if (!text.includes('/')) {
    return text
  }

  return text
    .split(ANGLE_SPAN_SPLIT_RE)
    .map((part, index) => (index % 2 === 1 ? part : linkifySegment(part)))
    .join('')
}
