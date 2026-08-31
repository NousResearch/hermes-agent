import { normalize } from '@/lib/text'

const VALID_LANGUAGE_RE = /^[a-z0-9][a-z0-9+#-]*$/i
const NON_CODE_FENCE_LANGUAGES = new Set(['', 'text', 'plain', 'plaintext', 'md', 'markdown'])

const COMMON_CODE_LANGUAGES = new Set([
  'bash',
  'c',
  'cpp',
  'css',
  'diff',
  'go',
  'html',
  'java',
  'javascript',
  'js',
  'json',
  'jsx',
  'markdown',
  'md',
  'php',
  'python',
  'py',
  'ruby',
  'rust',
  'rs',
  'sh',
  'sql',
  'swift',
  'tsx',
  'ts',
  'typescript',
  'xml',
  'yaml',
  'yml'
])

interface CodeSignals {
  bulletLines: number
  codeSignals: number
  hasMarkdown: boolean
  proseLines: number
  trimmed: string
  urlLines: number
}

export function sanitizeLanguageTag(tag: string): string {
  const trimmed = tag.trim()
  const first = trimmed.split(/\s/, 1)[0] || ''

  return VALID_LANGUAGE_RE.test(first) && first.length <= 16 ? first.toLowerCase() : ''
}

// Sanitized language tag → codicon glyph. Anything not listed falls back to
// the generic `code` glyph, which matches what the tool-row icons use.
const CODICON_BY_LANGUAGE: Record<string, string> = {
  bash: 'terminal',
  cmd: 'terminal',
  console: 'terminal',
  fish: 'terminal',
  powershell: 'terminal',
  ps1: 'terminal',
  sh: 'terminal',
  shell: 'terminal',
  zsh: 'terminal',

  md: 'markdown',
  markdown: 'markdown',

  json: 'json',
  json5: 'json',

  ini: 'settings-gear',
  toml: 'settings-gear',
  yaml: 'settings-gear',
  yml: 'settings-gear',
  dotenv: 'settings-gear',
  env: 'settings-gear',

  graphql: 'database',
  gql: 'database',
  mysql: 'database',
  postgres: 'database',
  postgresql: 'database',
  sql: 'database',
  sqlite: 'database',

  diff: 'diff',
  patch: 'diff',

  css: 'symbol-color',
  less: 'symbol-color',
  sass: 'symbol-color',
  scss: 'symbol-color',
  svg: 'symbol-color',

  regex: 'regex',
  regexp: 'regex',

  curl: 'globe',
  http: 'globe',

  docker: 'package',
  dockerfile: 'package',

  mermaid: 'graph'
}

export function codiconForLanguage(language: string | undefined): string {
  return CODICON_BY_LANGUAGE[sanitizeLanguageTag(language || '')] || 'code'
}

// File extension → language tag, so a filename can resolve to the same icon a
// fenced code block of that language would get. Only extensions that map to a
// non-generic codicon need an entry; everything else falls through to `code`.
const LANGUAGE_BY_EXTENSION: Record<string, string> = {
  bash: 'bash',
  cfg: 'ini',
  conf: 'ini',
  css: 'css',
  dockerfile: 'dockerfile',
  env: 'env',
  gql: 'graphql',
  graphql: 'graphql',
  ini: 'ini',
  json: 'json',
  json5: 'json',
  less: 'less',
  markdown: 'markdown',
  md: 'markdown',
  mdx: 'markdown',
  mmd: 'mermaid',
  ps1: 'powershell',
  psql: 'sql',
  sass: 'sass',
  scss: 'scss',
  sh: 'bash',
  sql: 'sql',
  svg: 'svg',
  toml: 'toml',
  yaml: 'yaml',
  yml: 'yml',
  zsh: 'zsh'
}

// Pick an icon for a file path by its extension (or bare name like
// `Dockerfile`), reusing the language→codicon map so file-edit rows and code
// blocks share one visual vocabulary. Unknown / generic code files get `code`.
export function codiconForFilename(path: string | undefined): string {
  const token = filenameExtToken(path)
  const language = LANGUAGE_BY_EXTENSION[token] || token

  return codiconForLanguage(language)
}

// Last path segment's extension (or the bare lowercased name for `Dockerfile`,
// `Makefile`, …). Shared by the icon and Shiki-language resolvers.
function filenameExtToken(path: string | undefined): string {
  const base = normalize((path || '').replace(/\\/g, '/').split('/').pop())
  const dot = base.lastIndexOf('.')

  return dot > 0 ? base.slice(dot + 1) : base
}

// File extension → Shiki bundled-language id, for syntax-highlighting diffs in
// the editing tool's own language. Unknown extensions return '' so callers fall
// back to the plain color-only diff renderer.
const SHIKI_LANGUAGE_BY_EXTENSION: Record<string, string> = {
  astro: 'astro',
  bash: 'bash',
  c: 'c',
  cc: 'cpp',
  cjs: 'javascript',
  clj: 'clojure',
  cpp: 'cpp',
  cs: 'csharp',
  css: 'css',
  cxx: 'cpp',
  dart: 'dart',
  dockerfile: 'docker',
  ex: 'elixir',
  exs: 'elixir',
  fish: 'fish',
  go: 'go',
  gql: 'graphql',
  graphql: 'graphql',
  h: 'c',
  hpp: 'cpp',
  hs: 'haskell',
  htm: 'html',
  html: 'html',
  ini: 'ini',
  java: 'java',
  jl: 'julia',
  js: 'javascript',
  json: 'json',
  json5: 'json5',
  jsonc: 'jsonc',
  jsx: 'jsx',
  kt: 'kotlin',
  kts: 'kotlin',
  less: 'less',
  lua: 'lua',
  makefile: 'make',
  markdown: 'markdown',
  md: 'markdown',
  mdx: 'mdx',
  mjs: 'javascript',
  ml: 'ocaml',
  mts: 'typescript',
  nix: 'nix',
  php: 'php',
  pl: 'perl',
  proto: 'proto',
  ps1: 'powershell',
  py: 'python',
  pyi: 'python',
  r: 'r',
  rb: 'ruby',
  rs: 'rust',
  sass: 'sass',
  scala: 'scala',
  scss: 'scss',
  sh: 'bash',
  sql: 'sql',
  svelte: 'svelte',
  swift: 'swift',
  tf: 'terraform',
  toml: 'toml',
  ts: 'typescript',
  tsx: 'tsx',
  vue: 'vue',
  xml: 'xml',
  yaml: 'yaml',
  yml: 'yaml',
  zig: 'zig',
  zsh: 'bash'
}

export function shikiLanguageForFilename(path: string | undefined): string {
  return SHIKI_LANGUAGE_BY_EXTENSION[filenameExtToken(path)] || ''
}

function proseLineCount(body: string): number {
  return body.split('\n').filter(line => {
    const trimmed = line.trim()

    return Boolean(trimmed) && /^[A-Za-z0-9"'`*-]/.test(trimmed)
  }).length
}

const CODE_SIGNAL_RE = [
  /(^|\s)(const|let|var|function|class|import|export|return|if|for|while|switch)\b/gim,
  /=>|==|===|!=|!==|\{|\}|;|<\/?[a-z][^>]*>/gi,
  /^\s*(#include|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b/gim
]

function codeSignalCount(body: string): number {
  return CODE_SIGNAL_RE.reduce((total, pattern) => total + (body.match(pattern)?.length ?? 0), 0)
}

function codeSignals(body: string): CodeSignals {
  const trimmed = body.trim()
  const markdownSignals = (trimmed.match(/\*\*[^*]+\*\*/g) || []).length + (trimmed.match(/`[^`\n]+`/g) || []).length

  return {
    bulletLines: (trimmed.match(/^\s*[-*]\s+\S+/gm) || []).length,
    codeSignals: codeSignalCount(trimmed),
    hasMarkdown: markdownSignals > 0,
    proseLines: proseLineCount(trimmed),
    trimmed,
    urlLines: (trimmed.match(/^\s*https?:\/\/\S+\s*$/gim) || []).length
  }
}

// A complete Markdown document with YAML frontmatter is structured text even
// when the frontmatter contains list-valued fields. Without this signal, the
// generic bullet-list heuristic below unwraps note previews such as inventory
// records, causing their YAML and body to be rendered as ordinary Markdown.
function hasYamlFrontmatter(body: string): boolean {
  const lines = body.trim().split('\n')

  if (lines[0]?.trim() !== '---') {
    return false
  }

  const closingIndex = lines.findIndex((line, index) => index > 0 && line.trim() === '---')

  if (closingIndex < 2) {
    return false
  }

  return lines.slice(1, closingIndex).some(line => CONFIG_SEPARATOR_LINE_RE.test(line.trim()))
}

// A sentence-ending punctuation mark followed by whitespace or end-of-line.
// Real wrapped prose has these; config/structured listings almost never do.
const SENTENCE_PUNCTUATION_RE = /[.!?](?:\s|$)/
// `Key: value` / `Key = value` — a settings/directive line with an explicit
// separator. Unambiguous config shape.
const CONFIG_SEPARATOR_LINE_RE = /^[A-Za-z0-9_][\w.-]*\s*[:=]\s*\S/
// A bare identifier that could be a config key (`Host`, `Port`, `HostName`,
// `API_KEY`). Used only for the short `Key value` directive form below.
const CONFIG_KEY_RE = /^[A-Za-z0-9_][\w.-]*$/

// True when a single line looks like a config directive rather than prose.
// Either an explicit `Key: value` / `Key = value`, or a short whitespace
// directive of 2-3 tokens led by an identifier (`Host example`, `Port 22`,
// `HostName 10.0.0.1`). The token cap is what separates it from prose — a
// real sentence line has more words than a config directive, so a
// punctuation-less prose fragment like `the quick brown fox jumps` (5 tokens)
// is NOT treated as config.
function isConfigDirectiveLine(line: string): boolean {
  const trimmed = line.trim()

  if (CONFIG_SEPARATOR_LINE_RE.test(trimmed)) {
    return true
  }

  const tokens = trimmed.split(/\s+/)

  return tokens.length >= 2 && tokens.length <= 3 && CONFIG_KEY_RE.test(tokens[0])
}

/**
 * True when a fenced block looks like structured / config / tabular text
 * rather than wrapped prose. Such blocks (SSH config, .env dumps, INI-style
 * settings, key/value listings) trip the prose heuristics' "3+ plain lines,
 * no JS/SQL tokens" rule and get their fence stripped — rendering as a flat
 * paragraph instead of a code block. This veto keeps them fenced.
 *
 * Two signals, biased toward keeping the fence when ambiguous:
 *   - ANY indented continuation line (leading whitespace before content):
 *     prose is not indented line-by-line, but config stanzas are
 *     (`Host x` / `    HostName y`).
 *   - No sentence-ending punctuation AND a majority of lines are
 *     `Key value` / `Key: value` directives (see isConfigDirectiveLine).
 */
export function isLikelyStructuredText(body: string): boolean {
  const lines = body.split('\n').filter(line => line.trim())

  if (lines.length < 2) {
    return false
  }

  if (lines.some(line => /^\s+\S/.test(line))) {
    return true
  }

  if (lines.some(line => SENTENCE_PUNCTUATION_RE.test(line.trim()))) {
    return false
  }

  const configLines = lines.filter(line => isConfigDirectiveLine(line)).length

  return configLines >= Math.max(2, Math.ceil(lines.length * 0.6))
}

export function isLikelyProseFence(info: string, body: string): boolean {
  const trimmedInfo = info.trim()
  const rawInfo = trimmedInfo.toLowerCase()
  const language = sanitizeLanguageTag(info)
  const infoToken = trimmedInfo.split(/\s+/, 1)[0] || ''
  const hasInfoTail = Boolean(trimmedInfo) && trimmedInfo !== infoToken

  if (/^[-*+]\s/.test(rawInfo) || /^https?:\/\//.test(rawInfo)) {
    return true
  }

  const signals = codeSignals(body)

  if (!signals.trimmed) {
    return false
  }

  if (
    hasInfoTail &&
    signals.codeSignals <= 2 &&
    (signals.proseLines >= 2 || signals.bulletLines >= 1 || signals.urlLines >= 1)
  ) {
    return true
  }

  if (!NON_CODE_FENCE_LANGUAGES.has(language)) {
    return false
  }

  // Config / key-value / indented listings are not prose — keep their fence
  // so an SSH config, .env, or INI block renders as a code block instead of
  // being unwrapped into a paragraph.
  if (isLikelyStructuredText(body)) {
    return false
  }

  return (
    (signals.bulletLines >= 2 && signals.hasMarkdown && signals.codeSignals <= 2) ||
    (signals.proseLines >= 3 && signals.codeSignals === 0)
  )
}

export function isLikelyProseCodeBlock(language: string | undefined, code: string | undefined): boolean {
  const cleanLanguage = sanitizeLanguageTag(language || '')
  const signals = codeSignals(code || '')

  if (!signals.trimmed || signals.codeSignals >= 3) {
    return false
  }

  if (hasYamlFrontmatter(signals.trimmed)) {
    return false
  }

  // A bullet list with markdown emphasis is prose even when it happens to be
  // structured; the config veto below is only meant to protect config/kv
  // listings, so let the bullet-prose case win first.
  if (signals.bulletLines >= 1 && (signals.hasMarkdown || signals.proseLines >= 2)) {
    return true
  }

  // Config / key-value / indented listings are code, not prose — never
  // unwrap them (SSH config, .env, INI, key/value tables).
  if (isLikelyStructuredText(code || '')) {
    return false
  }

  if (NON_CODE_FENCE_LANGUAGES.has(cleanLanguage)) {
    return signals.proseLines >= 3 && signals.codeSignals === 0
  }

  return !COMMON_CODE_LANGUAGES.has(cleanLanguage) && signals.proseLines >= 2 && signals.codeSignals <= 1
}
