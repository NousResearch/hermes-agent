/**
 * Pure helpers for BARE filesystem paths and URLs in assistant prose.
 *
 * Agents constantly answer with a naked path — "Report: /Users/me/report.md",
 * or `~/notes.md` in a code span — and until now that text was decoration:
 * copyable, never clickable. #89472 opened the `#preview/…` door for paths
 * the agent wrapped in a markdown link; this module finds the paths the agent
 * did NOT wrap so they can take the same door.
 *
 * Kept free of React/store imports so `preprocessMarkdown` (a hot per-flush
 * path) can call it, like `session-refs.ts`.
 */

import { isFileMediaPath } from '@/lib/media'

/** File extensions worth turning into a link. Deliberately a list, not "any
 *  dot": a bare `/usr/bin` or `a/b` in prose is more often a directory, a
 *  command or a ratio than a file to preview. */
const LINKABLE_EXTENSIONS = [
  'c',
  'cjs',
  'conf',
  'cpp',
  'css',
  'csv',
  'go',
  'h',
  'hpp',
  'htm',
  'html',
  'ini',
  'java',
  'js',
  'json',
  'jsonl',
  'jsx',
  'log',
  'lua',
  'md',
  'markdown',
  'mdx',
  'mjs',
  'pdf',
  'py',
  'rb',
  'rs',
  'sh',
  'sql',
  'svg',
  'toml',
  'ts',
  'tsx',
  'txt',
  'xml',
  'yaml',
  'yml',
  'zsh',
  // media — these route to `#media:` and keep their inline player
  'png',
  'jpg',
  'jpeg',
  'gif',
  'webp',
  'mp3',
  'mp4',
  'mov',
  'wav',
  'webm'
] as const

const EXT_ALT = LINKABLE_EXTENSIONS.join('|')

/**
 * A bare absolute path in prose: `/abs/…`, `~/…`, or `file://…`, ending in a
 * known extension. Segments may not contain whitespace or markdown-significant
 * punctuation, so a path never swallows the end of a sentence or a closing
 * paren. Windows drive paths are excluded on purpose: `C:\Users` reads the
 * backslash as an escape in markdown and needs its own treatment.
 *
 * Lookbehind rejects a match that is already the target of a markdown link
 * (`](/x.md)`), sits inside an autolink (`<file:///x>`), or is glued to a
 * word (`foo/bar.md` is relative — not ours).
 */
const BARE_PATH_RE = new RegExp(
  String.raw`(?<![\w\]\(<:/])(?:file:\/\/|~(?=\/)|(?=\/))[^\s\]\)<>"'\x60*]*?\.(?:${EXT_ALT})(?![\w\-/])(?!\.\w)`,
  'gi'
)

/** Punctuation an agent leaves glued to the end of a path in a sentence. */
const TRAILING_PUNCTUATION_RE = /[,.;:!?]+$/

/**
 * A single path or URL that fills an inline code span exactly. This is the
 * agent's favourite shape (`` `/tmp/report.md` ``) and the one the user
 * cannot click today. Stricter than prose: the WHOLE span must be the target.
 */
const CODE_SPAN_URL_RE = /^https?:\/\/[^\s<>"'`]+$/i
const CODE_SPAN_PATH_RE = /^(?:file:\/\/|~\/|\/)[^\s<>"'`]+$/i

export interface PathRefMatch {
  /** Trailing prose punctuation the greedy match picked up, re-emitted after the link. */
  trailing: string
  /** The path as written (may be `file://…` or `~/…`). */
  value: string
}

/** Strips sentence punctuation that is not part of the path. A path ending in
 *  `.md.` is a sentence end, not a file called `report.md.`. */
export function splitBarePath(raw: string): PathRefMatch {
  const value = raw.replace(TRAILING_PUNCTUATION_RE, '')

  return { trailing: raw.slice(value.length), value }
}

/** True for a bare `/abs/file.ext`, `~/file.ext` or `file:///…` with a
 *  known extension — the shape `linkifyBarePaths` rewrites. */
export function isBarePath(value: string): boolean {
  BARE_PATH_RE.lastIndex = 0

  const match = BARE_PATH_RE.exec(value)

  return match !== null && match.index === 0 && match[0].length === value.length
}

/**
 * Classifies the text of an inline code span. `'url'` for an http(s) URL,
 * `'path'` for an absolute/home/file: path, `null` for everything else (a
 * command, a symbol, a relative path). Relative paths stay inert: without a
 * cwd we cannot know what they point at, and `a/b` is too often not a file.
 */
export function classifyCodeSpan(text: string): 'path' | 'url' | null {
  const trimmed = text.trim()

  if (!trimmed || /\s/.test(trimmed)) {
    return null
  }

  if (CODE_SPAN_URL_RE.test(trimmed)) {
    return 'url'
  }

  if (CODE_SPAN_PATH_RE.test(trimmed) && isFileMediaPath(trimmed)) {
    return 'path'
  }

  return null
}

/** The href door a bare path takes: `#path/<encoded>`. Separate from
 *  `#preview/` on purpose — that door renders the block PreviewAttachment
 *  card for links the agent AUTHORED (#89472); a path the agent merely
 *  mentioned mid-sentence renders as an inline link (`InlinePathLink`), so
 *  the sentence stays a sentence. Fragment hrefs survive markdown
 *  sanitisation, same trick as `#session/`. */
export function pathMarkdownHref(value: string): string {
  return `#path/${encodeURIComponent(value)}`
}

export function pathFromMarkdownHref(href?: string): null | string {
  if (!href?.startsWith('#path/')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#path/'.length)) || null
  } catch {
    return null
  }
}

function displayLabel(value: string): string {
  return value.replace(/[[\]\\]/g, '\\$&')
}

/**
 * Rewrites bare absolute paths in PROSE into `#path/` markdown links so they
 * reach `MarkdownLink` and render as an inline link that opens the preview.
 * Callers must exclude code spans and fences — `preprocessMarkdown` already
 * does; code spans get their own treatment in the `inlineCode` renderer.
 *
 * The label is the path as written: the agent chose to show the path, so the
 * user still sees it (and can select/copy it), it just also opens now.
 */
export function linkifyBarePaths(text: string): string {
  if (!text.includes('/')) {
    return text
  }

  return text.replace(BARE_PATH_RE, (match: string) => {
    const { trailing, value } = splitBarePath(match)

    if (!value || !isFileMediaPath(value)) {
      return match
    }

    return `[${displayLabel(value)}](${pathMarkdownHref(value)})${trailing}`
  })
}
