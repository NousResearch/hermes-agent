import { isLikelyProseCodeBlock, sanitizeLanguageTag } from '@/lib/markdown-code'

/**
 * Artifact detection — decides when a fenced block in an assistant message is
 * substantial, self-contained content that deserves an artifact card (opening
 * in the right rail) instead of an inline code block.
 *
 * Pure and cheap: it runs per streaming delta on the growing fence body, so
 * everything here is a bounded regex scan or a line count. No store access.
 */

export type ArtifactKind = 'code' | 'html' | 'svg'

export interface ArtifactDetection {
  kind: ArtifactKind
  /** Sanitized fence language ('' possible for html/svg detected by shape). */
  language: string
  /** Human title derived from the content (html <title>, svg <title>, a named
   *  declaration for code). Falls back to a kind/language label. */
  title: string
}

// A fence only becomes an artifact once it is unambiguously a document (html),
// a large standalone graphic (svg), or long enough that inlining it would
// drown the conversation (code). Small snippets stay inline code cards.
const HTML_DOC_RE = /<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>]/i
const HTML_TAG_RE = /<[a-z][a-z0-9-]*(\s[^>]*)?>/i
const HTML_DOC_MIN_CHARS = 160
const HTML_FRAGMENT_MIN_CHARS = 1200
const SVG_MIN_CHARS = 2000
const CODE_MIN_LINES = 48
const CODE_MIN_CHARS = 3000

const HTML_LANGUAGES = new Set(['html', 'htm', 'xhtml'])

// Languages whose fences are never artifacts: prose-ish, terminal output, and
// the fences already owned by richer renderers (mermaid diagrams, small svg).
const NON_ARTIFACT_LANGUAGES = new Set([
  '',
  'console',
  'diff',
  'log',
  'logs',
  'markdown',
  'md',
  'mermaid',
  'output',
  'patch',
  'plain',
  'plaintext',
  'shell-session',
  'stdout',
  'text',
  'txt'
])

function countLines(text: string): number {
  let lines = 1
  let index = text.indexOf('\n')

  while (index !== -1) {
    lines += 1
    index = text.indexOf('\n', index + 1)
  }

  return lines
}

function stripTags(value: string): string {
  return value
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function titleFromTag(content: string, tag: 'h1' | 'title'): string {
  const match = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, 'i').exec(content)

  return match ? stripTags(match[1] || '').slice(0, 80) : ''
}

const CODE_DECLARATION_RE =
  /(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|struct|interface|enum|trait|impl|def|fn)\s+([A-Za-z_$][\w$]*)/

// `// app.py`, `# server.ts`, `<!-- index.html -->`, `/* main.rs */` on the
// first meaningful line — the de-facto LLM convention for naming a file.
const FILENAME_COMMENT_RE = /^\s*(?:\/\/|#|--|<!--|\/\*)\s*([\w./-]+\.[a-z0-9]{1,8})\b/i

function codeTitle(language: string, content: string): string {
  const head = content.slice(0, 2000)
  const fileName = FILENAME_COMMENT_RE.exec(head)?.[1]

  if (fileName) {
    return fileName
  }

  const declaration = CODE_DECLARATION_RE.exec(head)?.[1]

  if (declaration) {
    return declaration
  }

  return language
}

export function detectArtifact(language: string | undefined, code: string | undefined): ArtifactDetection | null {
  const trimmed = (code ?? '').trim()

  if (!trimmed) {
    return null
  }

  const clean = sanitizeLanguageTag(language || '')

  if (HTML_LANGUAGES.has(clean)) {
    const isDocument = HTML_DOC_RE.test(trimmed)

    if (
      (isDocument && trimmed.length >= HTML_DOC_MIN_CHARS) ||
      (!isDocument && trimmed.length >= HTML_FRAGMENT_MIN_CHARS && HTML_TAG_RE.test(trimmed))
    ) {
      return {
        kind: 'html',
        language: clean,
        title: titleFromTag(trimmed, 'title') || titleFromTag(trimmed, 'h1') || 'HTML'
      }
    }

    return null
  }

  if (clean === 'svg') {
    // Small svg fences keep their inline embed (svg-embed.tsx); only a large
    // standalone graphic graduates into an artifact tab.
    if (trimmed.length >= SVG_MIN_CHARS && /<svg[\s>]/i.test(trimmed)) {
      return { kind: 'svg', language: clean, title: titleFromTag(trimmed, 'title') || 'SVG' }
    }

    return null
  }

  if (NON_ARTIFACT_LANGUAGES.has(clean)) {
    return null
  }

  if (trimmed.length < CODE_MIN_CHARS && countLines(trimmed) < CODE_MIN_LINES) {
    return null
  }

  if (isLikelyProseCodeBlock(clean, trimmed)) {
    return null
  }

  return { kind: 'code', language: clean, title: codeTitle(clean, trimmed) }
}

/** Stable identity slug for versioning: the same (kind, title, language) in a
 *  session is treated as one artifact the model iterates on. */
export function artifactSlug(detection: Pick<ArtifactDetection, 'kind' | 'language' | 'title'>): string {
  const title = detection.title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)

  return `${detection.kind}:${detection.language}:${title || 'untitled'}`
}

/** Tiny non-cryptographic content hash (FNV-1a) for version dedupe. */
export function artifactContentHash(content: string): string {
  let hash = 0x811c9dc5

  for (let i = 0; i < content.length; i += 1) {
    hash ^= content.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }

  return (hash >>> 0).toString(36)
}

const DOWNLOAD_EXTENSION_BY_LANGUAGE: Record<string, string> = {
  bash: '.sh',
  c: '.c',
  cpp: '.cpp',
  csharp: '.cs',
  css: '.css',
  go: '.go',
  htm: '.html',
  html: '.html',
  java: '.java',
  javascript: '.js',
  js: '.js',
  json: '.json',
  jsx: '.jsx',
  kotlin: '.kt',
  php: '.php',
  py: '.py',
  python: '.py',
  rb: '.rb',
  rs: '.rs',
  ruby: '.rb',
  rust: '.rs',
  sh: '.sh',
  sql: '.sql',
  svg: '.svg',
  swift: '.swift',
  toml: '.toml',
  ts: '.ts',
  tsx: '.tsx',
  typescript: '.ts',
  xhtml: '.html',
  xml: '.xml',
  yaml: '.yaml',
  yml: '.yaml'
}

export function artifactDownloadName(kind: ArtifactKind, language: string, title: string): string {
  const base =
    title
      .replace(/[^\p{L}\p{N}._ -]+/gu, '')
      .trim()
      .replace(/\s+/g, '-')
      .slice(0, 60) || 'artifact'

  if (/\.[a-z0-9]{1,8}$/i.test(base)) {
    return base
  }

  const ext = kind === 'html' ? '.html' : kind === 'svg' ? '.svg' : DOWNLOAD_EXTENSION_BY_LANGUAGE[language] || '.txt'

  return `${base}${ext}`
}
